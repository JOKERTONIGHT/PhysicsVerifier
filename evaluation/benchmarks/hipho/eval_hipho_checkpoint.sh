#!/usr/bin/env bash
# Evaluate one HiPhO matrix label on a dedicated GPU/port.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
LABEL="${1:?usage: $0 <label> [model_dir]}"
MODEL_DIR="${2:-}"
MATRIX_DIR="${MATRIX_DIR:-${ROOT}/results/hipho_baseline_matrix_30b}"
CKPT_ROOT="${CKPT_ROOT:-/slow_share/jinjianhan/ckpt/qwen3-30b-physics-openrlhf/ckpt}"
BASE_MODEL="${BASE_MODEL:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
PORT="${PORT:-8767}"
CUDA_DEVICE="${CUDA_DEVICE:-5}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
SERVED_NAME="${SERVED_NAME:-qwen3-30b-a3b-instruct-2507}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_TOKENS="${MAX_TOKENS:-8192}"

case "${LABEL}" in
  base_30b) MODEL_DIR="${MODEL_DIR:-${BASE_MODEL}}" ;;
  global_step5) MODEL_DIR="${MODEL_DIR:-${CKPT_ROOT}/global_step5_hf}" ;;
  global_step10) MODEL_DIR="${MODEL_DIR:-${CKPT_ROOT}/global_step10_hf}" ;;
  *) echo "[error] unknown label ${LABEL}" >&2; exit 2 ;;
esac

OUT_SUB="${MATRIX_DIR}/${LABEL}"
mkdir -p "${OUT_SUB}"

RUN_ID="${LABEL}" MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
  SERVED_NAME="${SERVED_NAME}" \
  bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop || true
sleep 2
RUN_ID="${LABEL}" MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
  SERVED_NAME="${SERVED_NAME}" VLLM_READY_SECS="${VLLM_READY_SECS:-3600}" \
  bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" start

START_TS="$(date -Iseconds)"
PRED_OUT="${OUT_SUB}/predictions.jsonl"
SCORE_OUT="${OUT_SUB}/scores.json"
"${PYTHON}" "${ROOT}/evaluation/benchmarks/hipho/generate_hipho_predictions.py" \
  --input "${BENCH_ROOT:-/slow_share/jinjianhan/workspace/benchmarks/hipho}/hipho_text_only.jsonl" \
  --output "${PRED_OUT}" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model "${SERVED_NAME}" \
  --temperature "${TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}"

"${ROOT}/.venv/bin/python" "${ROOT}/evaluation/benchmarks/hipho/score_hipho_predictions.py" \
  --predictions "${PRED_OUT}" \
  --output "${SCORE_OUT}" \
  --no-use-verifier

END_TS="$(date -Iseconds)"
python3 - <<PY >"${OUT_SUB}/run_meta.json"
import json
print(json.dumps({
  "label": "${LABEL}",
  "model_dir": "${MODEL_DIR}",
  "port": int("${PORT}"),
  "cuda_device": int("${CUDA_DEVICE}"),
  "start": "${START_TS}",
  "end": "${END_TS}",
}, ensure_ascii=False, indent=2))
PY

RUN_ID="${LABEL}" PORT="${PORT}" bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop || true
echo "[ok] ${LABEL} eval done -> ${SCORE_OUT}"
