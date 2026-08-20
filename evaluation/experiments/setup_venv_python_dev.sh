#!/usr/bin/env bash
# Install CPython 3.10 headers into the project tree (no sudo).
# pip cannot provide Python.h; this unpacks libpython3.10-dev locally for vLLM/Triton JIT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_ROOT="${LOCAL_ROOT:-$ROOT/.local/deb-root}"
BIN_DIR="${BIN_DIR:-$ROOT/.local/bin}"
DEB="${DEB:-/tmp/libpython3.10-dev_3.10.12-1~22.04.15_amd64.deb}"

if [[ ! -f "$DEB" ]]; then
  echo "[fetch] downloading libpython3.10-dev ..."
  (cd /tmp && apt-get download libpython3.10-dev)
  DEB="$(ls -1 /tmp/libpython3.10-dev_*.deb | tail -1)"
fi

mkdir -p "$LOCAL_ROOT" "$BIN_DIR"
dpkg-deb -x "$DEB" "$LOCAL_ROOT"

if [[ ! -f "$LOCAL_ROOT/usr/include/python3.10/Python.h" ]]; then
  echo "[error] Python.h not found after extracting $DEB" >&2
  exit 1
fi

chmod +x "$BIN_DIR/gcc"
echo "[ok] Python.h installed at $LOCAL_ROOT/usr/include/python3.10/Python.h"
echo "[ok] gcc wrapper: $BIN_DIR/gcc (prepends local include path for Triton)"
