#!/usr/bin/env bash
# After preflight passes: launch outcome_only GRPO waiter, then RFT-diag on GPU 7, then gate waiter.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PREFLIGHT="${PREFLIGHT_REPORT:-${ROOT}/logs/preflight_group_gradient.json}"
POLL="${POLL_SECS:-30}"
DIAG_GPU="${DIAG_GPU:-7}"

echo "[pipeline] waiting for ${PREFLIGHT}"
while [[ ! -f "${PREFLIGHT}" ]]; do
  sleep "${POLL}"
done
preflight_ok=0
if "${ROOT}/.venv/bin/python" - <<PY
import json, sys
from pathlib import Path
rep = json.loads(Path("${PREFLIGHT}").read_text())
print("[pipeline] preflight", json.dumps({k: rep.get(k) for k in ("mixed_rate","trunc_rate","boxed_rate","pass")}), flush=True)
sys.exit(0 if rep.get("pass") else 2)
PY
then
  preflight_ok=1
fi

mkdir -p "${ROOT}/logs"
if [[ "${preflight_ok}" -eq 1 ]]; then
  nohup bash "${ROOT}/training/swift/wait_and_launch_hybrid_grpo.sh" \
    >>"${ROOT}/logs/wait_and_launch_outcome_grpo.log" 2>&1 &
  echo $! >"${ROOT}/logs/wait_and_launch_outcome_grpo.pid"
  echo "[pipeline] GRPO waiter pid=$(cat "${ROOT}/logs/wait_and_launch_outcome_grpo.pid")"
else
  echo "[pipeline] preflight failed; skipping GRPO launch"
fi

# RFT collapse diagnosis on the preflight GPU after vLLM has been stopped.
CUDA_VISIBLE_DEVICES="${DIAG_GPU}" NPROC_PER_NODE=1 \
  nohup bash "${ROOT}/training/swift/run_rft_lora_diag.sh" \
  >>"${ROOT}/logs/rft_lora_diag.log" 2>&1 &
echo $! >"${ROOT}/logs/rft_lora_diag.pid"
echo "[pipeline] RFT diag pid=$(cat "${ROOT}/logs/rft_lora_diag.pid")"

# Gate waiter starts after diag releases the GPU: poll the diag pid, then watch GRPO ckpts.
(
  diag_pid="$(cat "${ROOT}/logs/rft_lora_diag.pid")"
  while kill -0 "${diag_pid}" 2>/dev/null; do
    sleep "${POLL}"
  done
  if [[ "${preflight_ok}" -eq 1 ]]; then
    echo "[pipeline] rft diag finished; starting GRPO gate waiter"
    CUDA_DEVICE="${DIAG_GPU}" NEED_CKPTS="${NEED_CKPTS:-1}" \
      bash "${ROOT}/training/swift/wait_and_gate_outcome_grpo.sh"
  else
    echo "[pipeline] rft diag finished; running RFT diag gate on LoRA ckpts"
    CUDA_DEVICE="${DIAG_GPU}" QWEN8B_RFT_CKPT="${QWEN8B_RFT_DIAG_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-rft-lora-diag}" \
      bash "${ROOT}/training/swift/run_rft_gate_check.sh"
  fi
) >>"${ROOT}/logs/wait_and_gate_outcome_grpo.log" 2>&1 &
echo $! >"${ROOT}/logs/wait_and_gate_outcome_grpo.pid"
echo "[pipeline] gate waiter pid=$(cat "${ROOT}/logs/wait_and_gate_outcome_grpo.pid")"
