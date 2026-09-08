#!/usr/bin/env bash
# Machine-local paths for outcome-only GRPO. Copy to train_env.sh and edit.
#   cp training/swift/train_env.example.sh training/swift/train_env.sh
# train_env.sh is gitignored. Do not put API keys here.

# Repo root (leave empty to auto-detect from this file).
# export PHYSICS_ROOT=/path/to/PhysicsVerifier

# Python: ms-swift training venv, OpenRLHF/vLLM python, project .venv
export SWIFT_VENV="${SWIFT_VENV:-/path/to/venv/swift_train}"
export ORHF_PYTHON="${ORHF_PYTHON:-/path/to/venv/openrlhf_train/bin/python}"
export VENV="${VENV:-${PHYSICS_ROOT}/.venv}"
export VENV_PY="${VENV_PY:-${VENV}/bin/python}"

# Weights, checkpoints, tmp
export QWEN8B_MODEL_DIR="${QWEN8B_MODEL_DIR:-/path/to/models/Qwen3-8B}"
export CKPT_ROOT="${CKPT_ROOT:-/path/to/ckpt}"
export SLOW_TMP_ROOT="${SLOW_TMP_ROOT:-/path/to/tmp}"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-${PHYSICS_ROOT}}"

# Optional. Empty is fine if nvcc is already on PATH.
# export CUDA_HOME=/usr/local/cuda

# GRPO world size. 2-GPU example: N_TRAIN_GPUS=2 GRAD_ACCUM=4
export N_TRAIN_GPUS="${N_TRAIN_GPUS:-4}"

# Derived ckpt dirs (override individually if needed)
export QWEN8B_OUTCOME_CKPT="${QWEN8B_OUTCOME_CKPT:-${CKPT_ROOT}/qwen3-8b-outcome-only-rl}"
export QWEN8B_SFT_CKPT="${QWEN8B_SFT_CKPT:-${CKPT_ROOT}/qwen3-8b-physics-sft}"
export QWEN8B_RFT_CKPT="${QWEN8B_RFT_CKPT:-${CKPT_ROOT}/qwen3-8b-physics-rft-lora}"
export QWEN8B_RFT_DIAG_CKPT="${QWEN8B_RFT_DIAG_CKPT:-${CKPT_ROOT}/qwen3-8b-physics-rft-lora-diag}"
