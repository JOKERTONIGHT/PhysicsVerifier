#!/usr/bin/env bash
# Fast heldout/HiPhO answer-acc probe for a single HF checkpoint.
set -euo pipefail
# shellcheck disable=SC1091
_ROOT_CAND="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PHYSICS_ROOT:-${_ROOT_CAND}}/training/swift/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
SCRIPT_DIR="${ROOT}/evaluation/benchmarks/hipho"
PYTHON="${ORHF_PYTHON}"
MODEL_DIR="${1:?usage: eval_heldout_fast.sh MODEL_DIR [out_dir]}"
OUT="${2:-${MODEL_DIR}/heldout_fast_eval}"
PORT="${PORT:-8766}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
HELDOUT="${HELDOUT_JSONL:-}"
if [[ -z "${HELDOUT}" ]]; then
  if [[ -s "${ROOT}/data/rl/heldout_eval_trusted.jsonl" ]]; then
    HELDOUT="${ROOT}/data/rl/heldout_eval_trusted.jsonl"
  else
    HELDOUT="${ROOT}/data/rl/heldout_eval.jsonl"
  fi
fi
MAX_SAMPLES="${MAX_SAMPLES:-0}"
N_SAMPLES="${N_SAMPLES:-4}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS="${MAX_TOKENS:-8192}"
MAX_LEN="${MAX_LEN:-16384}"
GPU_UTIL="${GPU_UTIL:-0.85}"
RESUME="${RESUME:-0}"
TOKENIZER="${TOKENIZER:-${QWEN8B_MODEL_DIR}}"
export VLLM_READY_SECS="${VLLM_READY_SECS:-1800}"
mkdir -p "${OUT}"
cleanup() {
  RUN_ID="heldout_fast" PORT="${PORT}" bash "${SCRIPT_DIR}/manage_eval_vllm.sh" stop || true
}
trap cleanup EXIT
if [[ "${RESUME}" != "1" ]]; then
  rm -f "${OUT}/heldout_predictions.jsonl"
fi
RUN_ID="heldout_fast" MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
  MAX_LEN="${MAX_LEN}" GPU_UTIL="${GPU_UTIL}" SERVED_NAME=qwen3-8b \
  TOKENIZER="${TOKENIZER}" \
  bash "${SCRIPT_DIR}/manage_eval_vllm.sh" start
gen_args=(
  --input "${HELDOUT}"
  --output "${OUT}/heldout_predictions.jsonl"
  --base-url "http://127.0.0.1:${PORT}/v1"
  --model qwen3-8b
  --max-samples "${MAX_SAMPLES}"
  --n-samples "${N_SAMPLES}"
  --temperature "${TEMPERATURE}"
  --max-tokens "${MAX_TOKENS}"
  --concurrency "${CONCURRENCY:-16}"
)
if [[ "${RESUME}" == "1" ]]; then
  gen_args+=(--resume)
fi
"${PYTHON}" "${SCRIPT_DIR}/generate_hipho_predictions.py" "${gen_args[@]}"
"${VENV_PY}" "${SCRIPT_DIR}/score_hipho_predictions.py" \
  --predictions "${OUT}/heldout_predictions.jsonl" \
  --output "${OUT}/heldout_scores.json" \
  --n-samples "${N_SAMPLES}" \
  --no-use-verifier
cat "${OUT}/heldout_scores.json"
