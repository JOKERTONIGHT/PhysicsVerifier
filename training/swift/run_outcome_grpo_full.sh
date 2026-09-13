#!/usr/bin/env bash
# Outcome-only GRPO full run: 400 steps, part-frac reward, 4 GPU, from base.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
export MODE="${MODE:-full}"
export PHYSICS_REWARD_MODE=outcome_only
export PHYSICS_REWARD_FUNC=outcome_only
export MAX_STEPS="${MAX_STEPS:-400}"
export SAVE_STEPS="${SAVE_STEPS:-50}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-4096}"
export MAX_LENGTH="${MAX_LENGTH:-8192}"
export PER_DEVICE_TRAIN_BS="${PER_DEVICE_TRAIN_BS:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export DEEPSPEED="${DEEPSPEED:-zero2}"
export OFFLOAD_MODEL=false
export OFFLOAD_OPTIMIZER=false
export VLLM_ENFORCE_EAGER=false
export SLEEP_LEVEL=1
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.45}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export OVERLONG_FILTER=false
export SKIP_CALIBRATION=1
export SKIP_SMOKE_GATE=1
if [[ -s "${ROOT}/data/rl/tiers/easy_prompts.jsonl" ]]; then
  export PROMPT_DATA="${PROMPT_DATA:-${ROOT}/data/rl/tiers/easy_prompts.jsonl}"
elif [[ -s "${ROOT}/data/rl/swift_prompts_hybrid_band.jsonl" ]]; then
  export PROMPT_DATA="${PROMPT_DATA:-${ROOT}/data/rl/swift_prompts_hybrid_band.jsonl}"
fi
echo "[full] outcome_only steps=${MAX_STEPS} prompts=${PROMPT_DATA:-} model=${QWEN8B_MODEL_DIR}"
exec bash "${ROOT}/training/swift/launch_hybrid_grpo_4gpu.sh"
