#!/usr/bin/env bash
# Run locked HiPhO eval for base 30B, global_step5, and global_step10.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
VENV="${VENV:-${ROOT}/.venv}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
BASE_MODEL="${BASE_MODEL:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
CKPT_ROOT="${CKPT_ROOT:-/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf/ckpt}"
ACTOR_DIR="${ACTOR_DIR:-${CKPT_ROOT}/_actor}"
MATRIX_DIR="${MATRIX_DIR:-${ROOT}/results/hipho_baseline_matrix_30b}"
PORT="${PORT:-8766}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
SERVED_NAME="${SERVED_NAME:-qwen3-30b-a3b-instruct-2507}"
USE_VERIFIER_SCORE="${USE_VERIFIER_SCORE:-0}"

mkdir -p "${MATRIX_DIR}"
bash "${ROOT}/evaluation/benchmarks/hipho/setup_hipho.sh"

declare -A MODEL_DIRS=(
  [base_30b]="${BASE_MODEL}"
  [global_step5]="${CKPT_ROOT}/global_step5_hf"
  [global_step10]="${CKPT_ROOT}/global_step10_hf"
)

for tag in global_step5 global_step10; do
  if [[ ! -f "${MODEL_DIRS[${tag}]}/model.safetensors.index.json" && ! -f "${MODEL_DIRS[${tag}]}/model.safetensors" ]]; then
    TAG="${tag}" OUT_DIR="${MODEL_DIRS[${tag}]}" \
      bash "${ROOT}/training/openrlhf/convert_openrlhf_ckpt_to_hf.sh"
  fi
done

MANIFEST="${MATRIX_DIR}/manifest.json"
python3 - <<PY >"${MANIFEST}"
import json, os
print(json.dumps({
  "temperature": float("${TEMPERATURE}"),
  "max_tokens": int("${MAX_TOKENS}"),
  "max_samples": int("${MAX_SAMPLES}"),
  "served_name": "${SERVED_NAME}",
  "use_verifier_score": bool(int("${USE_VERIFIER_SCORE}")),
  "models": {
    "base_30b": "${BASE_MODEL}",
    "global_step5": "${CKPT_ROOT}/global_step5_hf",
    "global_step10": "${CKPT_ROOT}/global_step10_hf",
  }
}, ensure_ascii=False, indent=2))
PY

for label in base_30b global_step5 global_step10; do
  if [[ -n "${ONLY_LABELS:-}" ]]; then
    case ",${ONLY_LABELS}," in
      *",${label},"*) ;;
      *) continue ;;
    esac
  fi
  OUT_SUB="${MATRIX_DIR}/${label}"
  mkdir -p "${OUT_SUB}"
  MODEL_DIR="${MODEL_DIRS[${label}]}"
  echo "[matrix] evaluating ${label} from ${MODEL_DIR}"

  RUN_ID="${label}" MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
    bash "${SCRIPT_DIR}/manage_eval_vllm.sh" stop || true
  sleep 3

  START_TS="$(date -Iseconds)"
  RUN_ID="${label}" MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
    SERVED_NAME="${SERVED_NAME}" \
    bash "${SCRIPT_DIR}/manage_eval_vllm.sh" start

  PRED_OUT="${OUT_SUB}/predictions.jsonl"
  SCORE_OUT="${OUT_SUB}/scores.json"
  "${PYTHON}" "${ROOT}/evaluation/benchmarks/hipho/generate_hipho_predictions.py" \
    --input "${BENCH_ROOT:-/slow_share/jinjianhan/workspace/benchmarks/hipho}/hipho_text_only.jsonl" \
    --output "${PRED_OUT}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model "${SERVED_NAME}" \
    --max-samples "${MAX_SAMPLES}" \
    --temperature "${TEMPERATURE}" \
    --max-tokens "${MAX_TOKENS}"

  if [[ "${USE_VERIFIER_SCORE}" == "1" ]]; then
    "${VENV}/bin/python" "${ROOT}/evaluation/benchmarks/hipho/score_hipho_predictions.py" \
      --predictions "${PRED_OUT}" \
      --output "${SCORE_OUT}" \
      --use-verifier
  else
    "${VENV}/bin/python" "${ROOT}/evaluation/benchmarks/hipho/score_hipho_predictions.py" \
      --predictions "${PRED_OUT}" \
      --output "${SCORE_OUT}" \
      --no-use-verifier
  fi

  END_TS="$(date -Iseconds)"
  python3 - <<PY >"${OUT_SUB}/run_meta.json"
import json
print(json.dumps({
  "label": "${label}",
  "model_dir": "${MODEL_DIR}",
  "start": "${START_TS}",
  "end": "${END_TS}",
  "predictions": "${PRED_OUT}",
  "scores": "${SCORE_OUT}",
}, ensure_ascii=False, indent=2))
PY

  RUN_ID="${label}" bash "${SCRIPT_DIR}/manage_eval_vllm.sh" stop || true
  sleep 5
done

"${VENV}/bin/python" "${ROOT}/evaluation/benchmarks/hipho/summarize_hipho_baseline.py" \
  --matrix-dir "${MATRIX_DIR}" \
  --base-label base_30b \
  --output "${MATRIX_DIR}/summary.json"

echo "[ok] HiPhO baseline matrix done: ${MATRIX_DIR}"
