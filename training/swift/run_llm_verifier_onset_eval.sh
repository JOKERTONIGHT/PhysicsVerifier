#!/usr/bin/env bash
# After 100-step llm_step GRPO: eval step0 + ckpts 20/40/60/80/100 on one idle GPU.
# Generation never sees gold; official scoring is a separate process.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
VENV="${VENV:-${ROOT}/.venv}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
CKPT="${QWEN8B_LLM_VERIFIER_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-deepseek-v4-flash-grpo-onset}"
BASE_MODEL="${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
OUT_DIR="${OUT_DIR:-${CKPT}/onset_eval}"
BENCH_ROOT="${BENCH_ROOT:-/slow_share/jinjianhan/workspace/benchmarks/hipho}"
HIPHO_JSONL="${HIPHO_JSONL:-${BENCH_ROOT}/hipho_text_only.jsonl}"
HIPHO_MANIFEST="${HIPHO_MANIFEST:-${BENCH_ROOT}/hipho_official_manifest.json}"
HIPHO_GOLD="${HIPHO_GOLD:-${HIPHO_JSONL}}"
HELDOUT_JSONL="${HELDOUT_JSONL:-${ROOT}/data/rl/heldout_eval.jsonl}"
TRAIN_PROMPTS="${TRAIN_PROMPTS:-${ROOT}/data/rl/swift_prompts_max2048.jsonl}"
PORT="${PORT:-8766}"
FREE_MIB="${FREE_MIB:-75000}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
MAX_LEN="${MAX_LEN:-8192}"
VLLM="${VLLM:-/data1/jinjianhan/venv/openrlhf_train/bin/vllm}"
TOKENIZER="${TOKENIZER:-${BASE_MODEL}}"
SERVED_NAME="${SERVED_NAME:-qwen3-8b}"
GEN_CONCURRENCY="${GEN_CONCURRENCY:-16}"
EVAL_STEPS="${EVAL_STEPS:-0 10 20 30 40 50 60 80 100}"
HIPHO_GRADER_MODEL="${HIPHO_GRADER_MODEL:-deepseek-v4-flash}"

mkdir -p "${OUT_DIR}/predictions" "${OUT_DIR}/scores"

python3 "${ROOT}/training/rl_data/audit_eval_leakage.py" \
  --train "${TRAIN_PROMPTS}" \
  --heldout "${HELDOUT_JSONL}" \
  --hipho "${HIPHO_JSONL}" \
  --manifest "${OUT_DIR}/leakage_manifest.json" \
  --fail-on-exact || { echo "[error] leakage audit failed" >&2; exit 2; }

if [[ ! -f "${HIPHO_MANIFEST}" ]]; then
  echo "[warn] official HiPhO manifest missing; run setup_hipho.sh. Scoring will not claim paper HiPhO."
fi

if [[ -n "${EVAL_GPU:-}" ]]; then
  eval_gpu="${EVAL_GPU}"
else
  probe="$(python3 "${ROOT}/training/openrlhf/gpu_bundle_utils.py" probe --train-only --n-train 1 --free-mib "${FREE_MIB}" --util-max 5)"
  ok="$(python3 -c 'import json,sys; print(int(json.loads(sys.stdin.read()).get("ok", False)))' <<<"${probe}")"
  if [[ "${ok}" != "1" ]]; then
    echo "[refuse] need 1 idle GPU for eval: ${probe}" >&2
    exit 2
  fi
  eval_gpu="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["train_gpus"][0])' <<<"${probe}")"
fi

echo "[eval] gpu=${eval_gpu} tokenizer=${TOKENIZER} steps=${EVAL_STEPS}"

find_ckpt() {
  local step="$1"
  local found
  found="$(ls -d "${CKPT}"/v*-*/checkpoint-"${step}" "${CKPT}"/checkpoint-"${step}" 2>/dev/null | tail -1 || true)"
  echo "${found}"
}

declare -A MODELS
MODELS[0]="${BASE_MODEL}"
for step in ${EVAL_STEPS}; do
  if [[ "${step}" == "0" ]]; then
    continue
  fi
  path="$(find_ckpt "${step}")"
  if [[ -n "${path}" && -f "${path}/config.json" ]]; then
    MODELS["${step}"]="${path}"
  fi
done

start_vllm() {
  local model_dir="$1"
  local tries
  stop_vllm
  sleep 5
  for tries in 1 2 3; do
    CUDA_VISIBLE_DEVICES="${eval_gpu}" VLLM_USE_V1=0 TOKENIZERS_PARALLELISM=false \
      nohup "${VLLM}" serve "${model_dir}" \
      --tokenizer "${TOKENIZER}" \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --served-model-name "${SERVED_NAME}" \
      --dtype bfloat16 \
      --max-model-len "${MAX_LEN}" \
      --gpu-memory-utilization 0.7 \
      --enforce-eager \
      --trust-remote-code \
      --disable-log-requests \
      >"${OUT_DIR}/vllm.log" 2>&1 &
    echo $! >"${OUT_DIR}/vllm.pid"
    for _ in $(seq 1 72); do
      if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "[eval] vLLM ready model=${model_dir} attempt=${tries}"
        return 0
      fi
      if [[ -f "${OUT_DIR}/vllm.pid" ]] && ! kill -0 "$(cat "${OUT_DIR}/vllm.pid")" 2>/dev/null; then
        break
      fi
      sleep 5
    done
    echo "[warn] vLLM start attempt ${tries} failed; see ${OUT_DIR}/vllm.log" >&2
    stop_vllm
    sleep 8
  done
  echo "[error] vLLM failed to start; see ${OUT_DIR}/vllm.log" >&2
  return 2
}

stop_vllm() {
  if [[ -f "${OUT_DIR}/vllm.pid" ]]; then
    local pid
    pid="$(cat "${OUT_DIR}/vllm.pid" 2>/dev/null || true)"
    if [[ -n "${pid}" ]]; then
      pkill -TERM -P "${pid}" 2>/dev/null || true
      kill -TERM "${pid}" 2>/dev/null || true
      sleep 2
      pkill -KILL -P "${pid}" 2>/dev/null || true
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  fi
}

trap stop_vllm EXIT

_pred_complete() {
  local pred="$1"
  local gold="$2"
  local need have
  [[ -s "${pred}" && -s "${gold}" ]] || return 1
  need="$(wc -l < "${gold}")"
  have="$(wc -l < "${pred}")"
  [[ "${have}" -ge "${need}" ]]
}

_score_complete() {
  local tag="$1"
  local hip="${OUT_DIR}/scores/${tag}_hipho_official.json"
  [[ -s "${OUT_DIR}/scores/${tag}_heldout.json" && -s "${hip}" ]] || return 1
  "${VENV}/bin/python" - "${hip}" "${HIPHO_GRADER_MODEL}" <<'PY'
import json, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
got = str(data.get("grader_model") or "")
want = sys.argv[2]
sys.exit(0 if want in got else 1)
PY
}

gen_and_score() {
  local step="$1"
  local model_dir="$2"
  local tag="step${step}"
  local pred_dir="${OUT_DIR}/predictions/${tag}"
  mkdir -p "${pred_dir}"
  echo "[eval] begin ${tag} model=${model_dir}"
  if _pred_complete "${pred_dir}/heldout_predictions.jsonl" "${HELDOUT_JSONL}" \
    && _pred_complete "${pred_dir}/hipho_predictions.jsonl" "${HIPHO_JSONL}"; then
    echo "[eval] skip generation for ${tag}; predictions already complete"
  else
    if ! start_vllm "${model_dir}"; then
      echo "[error] skip ${tag}: vLLM failed" >&2
      return 0
    fi
    "${PYTHON}" "${ROOT}/evaluation/benchmarks/hipho/generate_hipho_predictions.py" \
      --input "${HELDOUT_JSONL}" \
      --output "${pred_dir}/heldout_predictions.jsonl" \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --model "${SERVED_NAME}" \
      --temperature "${TEMPERATURE}" \
      --max-tokens "${MAX_TOKENS}" \
      --concurrency "${GEN_CONCURRENCY}" \
      --strip-gold \
      --resume
    "${PYTHON}" "${ROOT}/evaluation/benchmarks/hipho/generate_hipho_predictions.py" \
      --input "${HIPHO_JSONL}" \
      --output "${pred_dir}/hipho_predictions.jsonl" \
      --base-url "http://127.0.0.1:${PORT}/v1" \
      --model "${SERVED_NAME}" \
      --temperature "${TEMPERATURE}" \
      --max-tokens "${MAX_TOKENS}" \
      --concurrency "${GEN_CONCURRENCY}" \
      --strip-gold \
      --resume
    stop_vllm
  fi
  if _score_complete "${tag}"; then
    echo "[eval] skip scoring for ${tag}; scores already complete"
    return 0
  fi
  "${VENV}/bin/python" "${ROOT}/evaluation/benchmarks/hipho/score_hipho_predictions.py" \
    --predictions "${pred_dir}/heldout_predictions.jsonl" \
    --gold "${HELDOUT_JSONL}" \
    --output "${OUT_DIR}/scores/${tag}_heldout.json" || \
  "${VENV}/bin/python" "${ROOT}/evaluation/benchmarks/hipho/score_hipho_predictions.py" \
    --predictions "${pred_dir}/heldout_predictions.jsonl" \
    --output "${OUT_DIR}/scores/${tag}_heldout.json"
  "${VENV}/bin/python" "${ROOT}/evaluation/benchmarks/hipho/score_hipho_official.py" \
    --predictions "${pred_dir}/hipho_predictions.jsonl" \
    --gold "${HIPHO_GOLD}" \
    --manifest "${HIPHO_MANIFEST}" \
    --grader-model "${HIPHO_GRADER_MODEL}" \
    --output "${OUT_DIR}/scores/${tag}_hipho_official.json" \
    --audit-jsonl "${OUT_DIR}/scores/${tag}_hipho_criteria.jsonl" \
    --allow-non-official || true
}

for step in ${EVAL_STEPS}; do
  if [[ -n "${MODELS[${step}]+x}" ]]; then
    gen_and_score "${step}" "${MODELS[${step}]}"
  fi
done

python3 - <<PY
import json
from pathlib import Path
out = Path("${OUT_DIR}")
steps = [int(x) for x in """${EVAL_STEPS}""".split()]
rows = []
for step in steps:
    held = out / "scores" / f"step{step}_heldout.json"
    hip = out / "scores" / f"step{step}_hipho_official.json"
    if not held.is_file():
        continue
    h = json.loads(held.read_text(encoding="utf-8"))
    hipho = json.loads(hip.read_text(encoding="utf-8")) if hip.is_file() else {}
    n = int(h.get("n_samples") or 0)
    acc = float(h.get("boxed_acc") or h.get("answer_acc") or 0.0)
    rows.append({
        "step": step,
        "heldout_n": n,
        "heldout_acc": acc,
        "heldout_correct": int(round(acc * n)),
        "hipho_mns": hipho.get("mns"),
        "hipho_total_points": hipho.get("total_points"),
        "hipho_full_marks": hipho.get("total_full_marks"),
        "official_reproduction": hipho.get("official_reproduction"),
        "grader_status": hipho.get("grader_status"),
        "boxed_acc": hipho.get("boxed_acc"),
    })
summary = {"checkpoints": rows}
(out / "onset_input.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

python3 "${ROOT}/training/swift/analyze_llm_verifier_onset.py" \
  --input "${OUT_DIR}/onset_input.json" \
  --output "${OUT_DIR}/onset_summary.json"

cp -f "${OUT_DIR}/scores/step100_hipho_official.json" "${OUT_DIR}/hipho_official_scores.json" 2>/dev/null || true
echo "[ok] onset eval written to ${OUT_DIR}"
