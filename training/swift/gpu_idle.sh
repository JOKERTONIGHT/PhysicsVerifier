#!/usr/bin/env bash
# Source this file: probe_idle_csv N [free_mib] [util_max]  → comma-separated GPU ids
_GPU_IDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${PHYSICS_ROOT:-}" || -z "${ORHF_PYTHON:-}" ]]; then
  # shellcheck disable=SC1091
  source "${_GPU_IDLE_DIR}/_load_train_env.sh"
fi
: "${ROOT:=${PHYSICS_ROOT}}"
: "${ORHF_PYTHON:?ORHF_PYTHON unset}"
: "${VENV_PY:=${ROOT}/.venv/bin/python}"

probe_idle_csv() {
  local n="${1:?n gpus}"
  local free="${2:-75000}"
  local util="${3:-5}"
  local probe ok ids
  probe="$("${ORHF_PYTHON}" "${ROOT}/training/openrlhf/gpu_bundle_utils.py" probe --train-only --n-train "${n}" --free-mib "${free}" --util-max "${util}")"
  ok="$("${VENV_PY}" -c 'import json,sys; print(int(json.loads(sys.stdin.read()).get("ok", False)))' <<<"${probe}")"
  ids="$("${VENV_PY}" -c 'import json,sys; d=json.loads(sys.stdin.read()); print(",".join(str(x) for x in d.get("train_gpus") or []))' <<<"${probe}")"
  if [[ "${ok}" != "1" || -z "${ids}" ]]; then
    echo "${probe}" >&2
    return 2
  fi
  echo "${ids}"
}

probe_one_gpu() {
  local csv
  csv="$(probe_idle_csv 1 "${1:-75000}" "${2:-5}")" || return 2
  echo "${csv%%,*}"
}

wait_idle_csv() {
  local n="${1:?n gpus}"
  local secs="${WAIT_GPU_SECS:-30}"
  local ids
  while true; do
    if ids="$(probe_idle_csv "${n}" "${2:-75000}" "${3:-5}" 2>/dev/null)"; then
      echo "${ids}"
      return 0
    fi
    sleep "${secs}"
  done
}
