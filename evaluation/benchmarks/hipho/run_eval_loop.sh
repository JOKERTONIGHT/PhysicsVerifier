#!/usr/bin/env bash
# Dual-track evaluation loop: HiPhO external bench + internal verifier metrics.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
CKPT_HF="${CKPT_HF:-}"
MODEL_NAME="${MODEL_NAME:-qwen3-30b-physics-rl}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8766/v1}"
HELDOUT="${HELDOUT:-${ROOT}/data/rl/heldout_eval.jsonl}"

if [[ -n "${CKPT_HF}" ]]; then
  echo "[eval] Converted HF checkpoint: ${CKPT_HF}"
  echo "[eval] Serve this checkpoint with vLLM before running HiPhO."
fi

bash "${ROOT}/evaluation/benchmarks/hipho/run_hipho_eval.sh"

if [[ -f "${HELDOUT}" ]]; then
  "${ROOT}/.venv/bin/python" "${ROOT}/scripts/run_physics_eval_pipeline.py" \
    --input "${HELDOUT}" \
    --output-dir "${ROOT}/results/eval_loop_internal" || true
fi

echo "[ok] eval loop finished"
