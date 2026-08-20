#!/usr/bin/env bash
# Move bulky data from /data1/jinjianhan and /home/jinjianhan to /slow_share,
# leaving symlinks at the original paths so running jobs keep working.
set -euo pipefail

SLOW_ROOT="${SLOW_ROOT:-/slow_share/jinjianhan}"
WS="${WORKSPACE_ROOT:-${SLOW_ROOT}/workspace}"
PHYSICS_ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
DATA1="${DATA1_ROOT:-/data1/jinjianhan}"

log() { echo "[migrate] $*"; }

move_dir() {
  local src="$1"
  local dest="$2"
  local link="$3"
  if [[ ! -e "${src}" ]]; then
    log "skip missing ${src}"
    return 0
  fi
  if [[ -L "${src}" ]]; then
    log "already symlink: ${src} -> $(readlink "${src}")"
    return 0
  fi
  mkdir -p "$(dirname "${dest}")"
  if [[ ! -d "${dest}" ]]; then
    log "rsync ${src} -> ${dest}"
    rsync -a "${src}/" "${dest}/"
  else
    log "dest exists, syncing delta ${src} -> ${dest}"
    rsync -a "${src}/" "${dest}/"
  fi
  local backup="${src}.bak.$(date +%Y%m%d%H%M%S)"
  log "rename ${src} -> ${backup}"
  mv "${src}" "${backup}"
  log "symlink ${link} -> ${dest}"
  ln -sfn "${dest}" "${link}"
}

mkdir -p "${WS}/benchmarks" "${WS}/models" "${WS}/catalogs" "${WS}/tmp" \
  "${SLOW_ROOT}/models"

log "=== migrate benchmarks ==="
move_dir "${DATA1}/benchmarks" "${WS}/benchmarks" "${DATA1}/benchmarks"

log "=== migrate tmp ==="
move_dir "${DATA1}/tmp" "${WS}/tmp" "${DATA1}/tmp"

log "=== migrate Qwen3-4B-AWQ weights ==="
if [[ -d "${DATA1}/models/qwen3_q4/Qwen3-4B-AWQ" && ! -L "${DATA1}/models/qwen3_q4/Qwen3-4B-AWQ" ]]; then
  move_dir "${DATA1}/models/qwen3_q4/Qwen3-4B-AWQ" \
    "${SLOW_ROOT}/models/Qwen3-4B-AWQ" \
    "${DATA1}/models/qwen3_q4/Qwen3-4B-AWQ"
fi

log "=== migrate catalogs ==="
if [[ -d "${PHYSICS_ROOT}/catalogs" && ! -L "${PHYSICS_ROOT}/catalogs" ]]; then
  move_dir "${PHYSICS_ROOT}/catalogs" "${WS}/catalogs" "${PHYSICS_ROOT}/catalogs"
fi

# venv stays on /data1 (local NVMe). NFS-backed venv slows Python imports and Ray worker startup.
log "=== skip venv (keep on data1 NVMe for training performance) ==="

log "=== done ==="
df -h / /data1 "${SLOW_ROOT}" | sed -n '1p;2p;3p;4p'
du -sh "${DATA1}" "${HOME}" "${WS}/venv"/* 2>/dev/null || true

log "=== optional: remove backup dirs after verifying symlinks ==="
for bak in "${DATA1}/benchmarks.bak."* "${DATA1}/tmp.bak."* \
  "${DATA1}/models/qwen3_q4/Qwen3-4B-AWQ.bak."* \
  "${PHYSICS_ROOT}/catalogs.bak."* \
  "${DATA1}/venv/PhysicsVerifier.bak."* "${DATA1}/venv/openrlhf_train.bak."*; do
  [[ -e "${bak}" ]] || continue
  log "backup kept (delete manually when verified): ${bak}"
done
