#!/usr/bin/env bash
# After hybrid 10-step pilot: fast heldout-50 + metric summary. Does not resume onset ckpt.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
CKPT="${QWEN8B_HYBRID_PILOT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-pilot10}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
# shellcheck disable=SC1091
source "${ROOT}/training/swift/gpu_idle.sh"

find_ckpt() {
  local found
  found="$(ls -d "${CKPT}"/v*-*/checkpoint-* "${CKPT}"/checkpoint-* 2>/dev/null | tail -1 || true)"
  echo "${found}"
}

latest="$(find_ckpt)"
heldout_scores="${CKPT}/heldout_fast_eval/heldout_scores.json"
if [[ -n "${latest}" && -f "${latest}/config.json" ]]; then
  echo "[observe] ckpt=${latest}"
  if [[ -z "${CUDA_DEVICE:-}" ]]; then
    CUDA_DEVICE="$(probe_one_gpu)" || { echo "[error] no idle GPU for heldout-50" >&2; exit 2; }
  fi
  CUDA_DEVICE="${CUDA_DEVICE}" PORT="${PORT:-8766}" MAX_SAMPLES="${MAX_SAMPLES:-50}" \
    bash "${ROOT}/training/swift/eval_heldout_fast.sh" "${latest}" "${CKPT}/heldout_fast_eval"
else
  echo "[observe] no checkpoint yet; summarizing metrics only"
fi

"${PYTHON}" "${ROOT}/training/swift/analyze_hybrid_pilot.py" \
  --ckpt "${CKPT}" \
  --metrics "${CKPT}/physics_reward_metrics.jsonl" \
  --heldout-scores "${heldout_scores}" \
  --output "${CKPT}/pilot_observe.json"
echo "[ok] wrote ${CKPT}/pilot_observe.json"
cat "${CKPT}/pilot_observe.json"
