#!/usr/bin/env bash
# Launch terra cascade audit of the 88 heldout gold labels.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
export SFT_API_MODEL="${SFT_API_MODEL:-gpt-5.6-terra}"
set +e
readarray -t _oa < <("${PYTHON}" "${ROOT}/training/rl_data/pick_sft_api.py" "${ROOT}")
_pick_rc=$?
set -e
if [[ "${_pick_rc}" -ne 0 || -z "${_oa[1]:-}" ]]; then
  echo "[audit] no usable API pair (rc=${_pick_rc})" >&2
  exit 2
fi
echo "[audit] using .env API pair ${_oa[3]:-?} model=${_oa[2]:-${SFT_API_MODEL}}"
OPENAI_BASE_URL="${_oa[0]}" OPENAI_API_KEY="${_oa[1]}" SFT_API_MODEL="${_oa[2]:-${SFT_API_MODEL}}" \
  "${PYTHON}" "${ROOT}/training/rl_data/audit_heldout_gold.py" \
    --api-base-url "${_oa[0]}" \
    --api-key "${_oa[1]}" \
    --api-model "${_oa[2]:-${SFT_API_MODEL}}" \
    --concurrency "${CONCURRENCY:-8}" \
    --max-tokens 4096 \
    --timeout 300
