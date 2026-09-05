#!/usr/bin/env bash
# Fill labeled SFT to TARGET_SFT_ROWS (~300) on a machine that has data/rl/.
# Gold hints are generation-only. Does not start 8B SFT/GRPO.
# Does not use GPUs 0–3 unless SFT_GEN_GPUS is set explicitly.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PYTHON="${PYTHON:-python3}"
TARGET="${TARGET_SFT_ROWS:-300}"
cd "${ROOT}"

if [[ ! -f "${ROOT}/data/rl/swift_prompts.jsonl" ]]; then
  echo "[error] missing data/rl/swift_prompts.jsonl (gitignored local dump)." >&2
  echo "        This fill must run on the GPU workstation that holds data/rl/." >&2
  exit 2
fi

mkdir -p "${ROOT}/data/rl" "${ROOT}/logs"

echo "[sft-fill] append hand-labeled complete stems"
"${PYTHON}" "${ROOT}/training/rl_data/sft_manual_fill.py" || true

echo "[sft-fill] rank unlabeled text-only candidates"
"${PYTHON}" "${ROOT}/training/rl_data/build_sft_fill_candidates.py"

n_now="$(wc -l < "${ROOT}/data/rl/sft_solutions.jsonl" | tr -d ' ' || echo 0)"
echo "[sft-fill] labeled now=${n_now} target=${TARGET}"
if [[ "${n_now}" -ge "${TARGET}" ]]; then
  echo "[sft-fill] already at target; audit only"
else
  if [[ "${SKIP_LOCAL:-0}" != "1" ]]; then
    if curl -sf --max-time 3 "http://127.0.0.1:${SFT_GEN_PORT:-8781}/v1/models" >/dev/null 2>&1; then
      BASE="http://127.0.0.1:${SFT_GEN_PORT:-8781}/v1"
    elif curl -sf --max-time 3 "http://127.0.0.1:8780/v1/models" >/dev/null 2>&1; then
      BASE="http://127.0.0.1:8780/v1"
    else
      echo "[error] no local 30B generator on :8781 or :8780. Start vLLM on an idle GPU first." >&2
      exit 3
    fi
    echo "[sft-fill] generate via ${BASE} until ${TARGET}"
    "${PYTHON}" "${ROOT}/training/rl_data/generate_sft_solutions.py" \
      --prompts "${ROOT}/data/rl/sft_fill_candidates.jsonl" \
      --output "${ROOT}/data/rl/sft_solutions.jsonl" \
      --unsolved "${ROOT}/data/rl/sft_unsolved_fill.jsonl" \
      --report "${ROOT}/data/rl/sft_fill_gen_report.json" \
      --base-url "${BASE}" \
      --model "${SFT_GEN_MODEL:-qwen3-30b-a3b}" \
      --local-only --hint-gold --k "${SFT_GEN_K:-2}" \
      --concurrency "${SFT_GEN_CONCURRENCY:-8}" \
      --target-solved "${TARGET}" \
      --fewshot "${ROOT}/training/rl_data/sft_fewshot.json"
  fi
fi

echo "[sft-fill] audit and drop visual/gold-fit/meta-talk"
"${PYTHON}" "${ROOT}/training/rl_data/audit_sft_solutions.py" \
  --input "${ROOT}/data/rl/sft_solutions.jsonl" \
  --apply || true

if [[ -f "${ROOT}/data/rl/sft_solutions.jsonl" ]]; then
  cp -f "${ROOT}/data/rl/sft_solutions.jsonl" "${ROOT}/data/rl/sft_solutions.manual.jsonl"
fi
n_final="$(wc -l < "${ROOT}/data/rl/sft_solutions.jsonl" | tr -d ' ' || echo 0)"
echo "[ok] labeled SFT rows=${n_final}"
