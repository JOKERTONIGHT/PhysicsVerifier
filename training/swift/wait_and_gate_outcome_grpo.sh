#!/usr/bin/env bash
# After outcome_only GRPO is running, wait for full checkpoints then run the heldout gate.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
CKPT="${1:-${QWEN8B_OUTCOME_CKPT}}"
POLL="${POLL_SECS:-60}"
NEED="${NEED_CKPTS:-1}"

RFT_PID_FILE="${RFT_GATE_PID_FILE:-${ROOT}/logs/rft_gate_check.pid}"
if [[ -f "${RFT_PID_FILE}" ]]; then
  rft_pid="$(cat "${RFT_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${rft_pid}" ]] && kill -0 "${rft_pid}" 2>/dev/null && [[ "${rft_pid}" != "$$" ]]; then
    echo "[gate-wait] waiting for RFT diag gate pid=${rft_pid} to release GPU ${CUDA_DEVICE:-0}"
    while kill -0 "${rft_pid}" 2>/dev/null; do
      sleep "${POLL}"
    done
  fi
fi

echo "[gate-wait] watching ${CKPT} for ${NEED} full checkpoint(s)"
while true; do
  n=0
  shopt -s nullglob
  for d in "${CKPT}"/v*-*/checkpoint-* "${CKPT}"/checkpoint-*; do
    [[ -d "${d}" ]] || continue
    [[ "${d}" == *-merged ]] && continue
    [[ -f "${d}/config.json" ]] || continue
    [[ -f "${d}/adapter_config.json" ]] && continue
    n=$((n + 1))
  done
  echo "[gate-wait] found ${n} checkpoint(s)"
  if [[ "${n}" -ge "${NEED}" ]]; then
    break
  fi
  sleep "${POLL}"
done
CUDA_DEVICE="${CUDA_DEVICE:-0}" PORT="${PORT:-8766}" \
  bash "${ROOT}/training/swift/run_outcome_grpo_gate.sh" "${CKPT}"
