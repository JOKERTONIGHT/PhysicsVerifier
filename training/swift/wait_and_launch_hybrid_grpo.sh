#!/usr/bin/env bash
# Wait for N idle GPUs then launch outcome_only GRPO from base Qwen3-8B.
# Requires a passing preflight report unless SKIP_PREFLIGHT=1.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
# shellcheck disable=SC1091
source "${ROOT}/training/swift/gpu_idle.sh"

BASE_MODEL="${QWEN8B_MODEL_DIR}"
PYTHON="${VENV_PY}"
PROMPTS="${PROMPT_DATA:-${ROOT}/data/rl/swift_prompts_hybrid_band.jsonl}"
PREFLIGHT="${PREFLIGHT_REPORT:-${ROOT}/logs/preflight_group_gradient.json}"
N_GPUS="${N_TRAIN_GPUS:-4}"

if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  if [[ ! -f "${PREFLIGHT}" ]]; then
    echo "[error] missing preflight ${PREFLIGHT}; run training/swift/run_preflight_band.sh first" >&2
    exit 2
  fi
  "${PYTHON}" - <<PY
import json, sys
from pathlib import Path
rep = json.loads(Path("${PREFLIGHT}").read_text())
if not rep.get("pass"):
    print("[error] preflight failed:", json.dumps(rep, indent=2))
    sys.exit(2)
print("[grpo-wait] preflight ok mixed={:.3f} trunc={:.3f} boxed_complete={:.3f}".format(
    float(rep.get("mixed_rate") or 0),
    float(rep.get("trunc_rate") or 0),
    float(rep.get("boxed_rate_complete") or rep.get("boxed_rate") or 0),
))
PY
fi

if [[ ! -s "${PROMPTS}" ]]; then
  echo "[error] missing banded prompts ${PROMPTS}" >&2
  exit 2
fi

echo "[grpo-wait] prompts=${PROMPTS} actor=${BASE_MODEL} reward=outcome_only n_gpus=${N_GPUS}"
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "[grpo-wait] waiting for ${N_GPUS} idle GPUs"
  wait_idle_csv "${N_GPUS}" >/dev/null
fi
export QWEN8B_MODEL_DIR="${BASE_MODEL}"
export PROMPT_DATA="${PROMPTS}"
export MODE=full
export SKIP_CALIBRATION=1
export SKIP_SMOKE_GATE=1
export PHYSICS_REWARD_MODE=outcome_only
export PHYSICS_REWARD_FUNC=outcome_only
export NPROC_PER_NODE="${N_GPUS}"
export PLUGIN="${ROOT}/training/swift/hybrid_reward_plugin.py"
bash "${ROOT}/training/swift/launch_hybrid_grpo_4gpu.sh"
echo "[ok] outcome_only GRPO launched from ${BASE_MODEL}"
mkdir -p "${ROOT}/logs"
nohup env CUDA_DEVICE="${GATE_CUDA_DEVICE:-${CUDA_DEVICE:-0}}" \
  bash "${ROOT}/training/swift/wait_and_gate_outcome_grpo.sh" \
  >>"${ROOT}/logs/wait_and_gate_grpo_outcome.log" 2>&1 &
echo $! >"${ROOT}/logs/wait_and_gate_grpo_outcome.pid"
echo "[grpo-wait] gate waiter pid=$(cat "${ROOT}/logs/wait_and_gate_grpo_outcome.pid")"
