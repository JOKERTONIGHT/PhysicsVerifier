#!/usr/bin/env bash
# Outcome-only vs hybrid ablation on the easy tier (same seed, same steps).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
export PROMPT_DATA="${PROMPT_DATA:-${ROOT}/data/rl/tiers/easy_prompts.jsonl}"
export MODE="${MODE:-pilot}"
export MAX_STEPS="${MAX_STEPS:-30}"
export SAVE_STEPS="${SAVE_STEPS:-10}"
export SKIP_CALIBRATION=1
export SKIP_SMOKE_GATE=1
export DEEPSPEED="${DEEPSPEED:-zero2}"
export OFFLOAD_MODEL=false
export OFFLOAD_OPTIMIZER=false
ARM="${ARM:-outcome}"
GATE="${ROOT}/logs/process_reward_gate.json"
if [[ "${ARM}" == "hybrid" ]]; then
  if [[ ! -f "${GATE}" ]] || ! "${VENV_PY}" -c 'import json,sys; g=json.load(open(sys.argv[1])); sys.exit(0 if (g.get("auc_error_eval") or {}).get("pass") else 2)' "${GATE}"; then
    echo "[refuse] hybrid ablation blocked: process AUC gate failed or missing. See ${GATE}" >&2
    exit 2
  fi
fi
if [[ "${ARM}" == "hybrid" ]]; then
  export PHYSICS_REWARD_MODE=hybrid_llm_outcome
  export PHYSICS_REWARD_FUNC=hybrid_llm_outcome
  export PHYSICS_REWARD_W_PROCESS="${PHYSICS_REWARD_W_PROCESS:-0.2}"
  export QWEN8B_HYBRID_PILOT_CKPT="${CKPT_ROOT}/qwen3-8b-easy-hybrid-ablation"
else
  export PHYSICS_REWARD_MODE=outcome_only
  export PHYSICS_REWARD_FUNC=outcome_only
  export QWEN8B_OUTCOME_PILOT_CKPT="${CKPT_ROOT}/qwen3-8b-easy-outcome-ablation"
fi
echo "[ablation] arm=${ARM} steps=${MAX_STEPS} prompts=${PROMPT_DATA}"
exec bash "${ROOT}/training/swift/launch_hybrid_grpo_4gpu.sh"
