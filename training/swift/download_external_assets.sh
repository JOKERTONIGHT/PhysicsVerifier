#!/usr/bin/env bash
# Download PHYSICS, UGPhysics, and Intern-S1-mini onto /slow_share via hf-mirror.
set -u
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/slow_share/jinjianhan/models/hf_cache}"
PYTHON="${PYTHON:-/home/jinjianhan/PhysicsVerifier/.venv/bin/python}"
LOG="${LOG:-/home/jinjianhan/PhysicsVerifier/logs/download_external_assets.log}"
mkdir -p "$(dirname "$LOG")" /slow_share/jinjianhan/datasets /slow_share/jinjianhan/models

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG"; }

download() {
  local repo="$1" dest="$2" kind="$3"
  mkdir -p "$dest"
  log "download ${kind} ${repo} -> ${dest}"
  "$PYTHON" - <<PY >>"$LOG" 2>&1
import os
repo = "${repo}"
dest = "${dest}"
kind = "${kind}"
try:
    from huggingface_hub import snapshot_download
    kwargs = {"repo_id": repo, "local_dir": dest, "local_dir_use_symlinks": False}
    if kind == "dataset":
        kwargs["repo_type"] = "dataset"
    snapshot_download(**kwargs)
    print("hf ok", repo)
except Exception as exc:
    print("hf failed", repo, exc)
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
        ms_download(repo, local_dir=dest)
        print("modelscope ok", repo)
    except Exception as exc2:
        print("modelscope failed", repo, exc2)
        raise SystemExit(1)
PY
}

# HF card is desimfj/PHYSICS (Zhengsh123/PHYSICS 404). Train split may still be Google Drive-only.
download "desimfj/PHYSICS" "/slow_share/jinjianhan/datasets/PHYSICS" dataset || true
download "UGPhysics/ugphysics" "/slow_share/jinjianhan/datasets/UGPhysics" dataset || true
download "internlm/Intern-S1-mini" "/slow_share/jinjianhan/models/Intern-S1-mini" model || true
log "done"
du -sh /slow_share/jinjianhan/datasets/PHYSICS /slow_share/jinjianhan/datasets/UGPhysics /slow_share/jinjianhan/models/Intern-S1-mini 2>/dev/null | tee -a "$LOG"
