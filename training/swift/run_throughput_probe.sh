#!/usr/bin/env bash
# 2-step throughput probe for the overhauled GRPO launch flags.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
export MODE=smoke
export PHYSICS_REWARD_MODE=outcome_only
export PHYSICS_REWARD_FUNC=outcome_only
export MAX_STEPS=2
export SAVE_STEPS=2
export SKIP_CALIBRATION=1
export SKIP_SMOKE_GATE=1
export DEEPSPEED="${DEEPSPEED:-zero2}"
export OFFLOAD_MODEL=false
export OFFLOAD_OPTIMIZER=false
export VLLM_ENFORCE_EAGER=false
export SLEEP_LEVEL=1
export VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.45}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
export MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-4096}"
export MAX_LENGTH="${MAX_LENGTH:-8192}"
echo "[probe] 2-step outcome_only smoke on ${QWEN8B_MODEL_DIR}"
exec bash "${ROOT}/training/swift/launch_hybrid_grpo_4gpu.sh"
