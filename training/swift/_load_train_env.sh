#!/usr/bin/env bash
# Source from any pipeline script. Resolves PHYSICS_ROOT, then loads train_env.sh.
# shellcheck disable=SC1091
_TRAIN_SWIFT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO_ROOT="$(cd "${_TRAIN_SWIFT_DIR}/../.." && pwd)"
export PHYSICS_ROOT="${PHYSICS_ROOT:-${_REPO_ROOT}}"
_ENV_FILE="${PHYSICS_ROOT}/training/swift/train_env.sh"
if [[ ! -f "${_ENV_FILE}" ]]; then
  echo "[error] missing ${_ENV_FILE}" >&2
  echo "Copy training/swift/train_env.example.sh to train_env.sh and edit paths." >&2
  return 2 2>/dev/null || exit 2
fi
# shellcheck disable=SC1090
source "${_ENV_FILE}"
export PHYSICS_ROOT="${PHYSICS_ROOT:-${_REPO_ROOT}}"
export VENV="${VENV:-${PHYSICS_ROOT}/.venv}"
export VENV_PY="${VENV_PY:-${VENV}/bin/python}"
export PYTHON="${PYTHON:-${VENV_PY}}"
_missing=()
for _v in SWIFT_VENV ORHF_PYTHON QWEN8B_MODEL_DIR CKPT_ROOT SLOW_TMP_ROOT; do
  if [[ -z "${!_v:-}" ]]; then
    _missing+=("${_v}")
  fi
done
if [[ ${#_missing[@]} -gt 0 ]]; then
  echo "[error] unset in train_env.sh: ${_missing[*]}" >&2
  return 2 2>/dev/null || exit 2
fi
export N_TRAIN_GPUS="${N_TRAIN_GPUS:-4}"
export QWEN8B_OUTCOME_CKPT="${QWEN8B_OUTCOME_CKPT:-${CKPT_ROOT}/qwen3-8b-outcome-only-rl}"
export QWEN8B_OUTCOME_SMOKE_CKPT="${QWEN8B_OUTCOME_SMOKE_CKPT:-${CKPT_ROOT}/qwen3-8b-outcome-only-smoke}"
export QWEN8B_OUTCOME_PILOT_CKPT="${QWEN8B_OUTCOME_PILOT_CKPT:-${CKPT_ROOT}/qwen3-8b-outcome-only-pilot}"
export QWEN8B_HYBRID_CKPT="${QWEN8B_HYBRID_CKPT:-${CKPT_ROOT}/qwen3-8b-hybrid-outcome-rl}"
export QWEN8B_HYBRID_SMOKE_CKPT="${QWEN8B_HYBRID_SMOKE_CKPT:-${CKPT_ROOT}/qwen3-8b-hybrid-outcome-smoke}"
export QWEN8B_HYBRID_PILOT_CKPT="${QWEN8B_HYBRID_PILOT_CKPT:-${CKPT_ROOT}/qwen3-8b-hybrid-outcome-pilot10}"
export QWEN8B_SFT_CKPT="${QWEN8B_SFT_CKPT:-${CKPT_ROOT}/qwen3-8b-physics-sft}"
export QWEN8B_RFT_CKPT="${QWEN8B_RFT_CKPT:-${CKPT_ROOT}/qwen3-8b-physics-rft-lora}"
export QWEN8B_RFT_DIAG_CKPT="${QWEN8B_RFT_DIAG_CKPT:-${CKPT_ROOT}/qwen3-8b-physics-rft-lora-diag}"
export WORKSPACE_ROOT="${WORKSPACE_ROOT:-${PHYSICS_ROOT}}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
unset _TRAIN_SWIFT_DIR _REPO_ROOT _ENV_FILE _v _missing
