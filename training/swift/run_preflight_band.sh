#!/usr/bin/env bash
# Stage 0: sample the 233-prompt difficulty band at GRPO settings, then gate mixed/trunc/boxed.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
PYTHON="${VENV_PY}"
PROMPTS="${PROMPTS:-${ROOT}/data/rl/swift_prompts_hybrid_band.jsonl}"
OUT="${OUT:-${ROOT}/data/rl/preflight_band_t10_3072.jsonl}"
SUMMARY="${SUMMARY:-${ROOT}/logs/preflight_band_summary.json}"
REPORT="${REPORT:-${ROOT}/logs/preflight_group_gradient.json}"
MODEL_DIR="${QWEN8B_MODEL_DIR}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PORT="${PORT:-8766}"
N_SAMPLES="${N_SAMPLES:-8}"
TEMPERATURE="${TEMPERATURE:-1.0}"
MAX_TOKENS="${MAX_TOKENS:-3072}"
MAX_LEN="${MAX_LEN:-16384}"

[[ -s "${PROMPTS}" ]] || { echo "[error] missing ${PROMPTS}" >&2; exit 2; }
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "[error] missing ${MODEL_DIR}" >&2; exit 2; }

RUN_ID="preflight_band" MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
  MAX_LEN="${MAX_LEN}" GPU_UTIL="${GPU_UTIL:-0.90}" SERVED_NAME=qwen3-8b \
  TOKENIZER="${MODEL_DIR}" \
  bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" start

set +e
"${PYTHON}" "${ROOT}/training/swift/rollout_pass_rates.py" \
  --prompts "${PROMPTS}" \
  --output "${OUT}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model qwen3-8b \
  --n-samples "${N_SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}" \
  --concurrency "${CONCURRENCY:-16}" \
  --summary "${SUMMARY}"
roll_rc=$?
set -e
RUN_ID="preflight_band" PORT="${PORT}" bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop || true
[[ "${roll_rc}" -eq 0 ]] || exit "${roll_rc}"

"${PYTHON}" "${ROOT}/training/swift/report_group_gradient.py" \
  --rollouts "${OUT}" \
  --output "${REPORT}"
echo "[preflight] report=${REPORT}"
