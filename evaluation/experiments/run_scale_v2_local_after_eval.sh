#!/usr/bin/env bash
# Wait for eval v2 upper-bound pipeline, then run leak-free scale curve on the v2 eval set
# with local 30B vLLM.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

EVAL_V2_TAG="${EVAL_V2_TAG:-eval_v2_deepseek_v4_pro_seed20260508}"
EVAL_V2_RESULTS="${EVAL_V2_RESULTS:-results/${EVAL_V2_TAG}}"
EVAL_V2_METRICS="${EVAL_V2_METRICS:-$EVAL_V2_RESULTS/upper_bound_error_metrics.json}"
EVAL_V2_DAEMON_PID="${EVAL_V2_DAEMON_PID:-$EVAL_V2_RESULTS/daemon.pid}"
EVAL_V2_DATASET="${EVAL_V2_DATASET:-data/derived/${EVAL_V2_TAG}/error_eval_dataset_100.json}"

SEED="${SEED:-20260508}"
DATA_DIR="${DATA_DIR:-data/derived/expansion3000_scale_v2_eval_deepseek_seed${SEED}}"
RESULT_ROOT="${RESULT_ROOT:-results/scale_curve_error_v2_local_30b}"
CATALOG_ROOT="${CATALOG_ROOT:-catalogs/scale_curve_error_v2_local_30b}"
REPORT_OUTPUT="${REPORT_OUTPUT:-docs/规则库规模曲线实验报告_v2_local_30b.md}"
STAMP_FILE="${STAMP_FILE:-$ROOT/results/_scale_v2_local_30b_stamp.txt}"

WAIT_POLL_SEC="${WAIT_POLL_SEC:-60}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-0}"  # 0 = no timeout

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

dataset_sample_count() {
  "$ROOT/.venv/bin/python" - <<PY
import json
from pathlib import Path
p = Path("$EVAL_V2_DATASET")
if not p.exists():
    print(0)
else:
    rows = json.loads(p.read_text(encoding="utf-8"))
    print(len(rows) if isinstance(rows, list) else 0)
PY
}

eval_v2_daemon_running() {
  if [[ ! -f "$EVAL_V2_DAEMON_PID" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$EVAL_V2_DAEMON_PID")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

wait_for_eval_v2() {
  log "waiting for eval v2 pipeline: metrics=$EVAL_V2_METRICS"
  local waited=0
  while [[ ! -f "$EVAL_V2_METRICS" ]]; do
    if eval_v2_daemon_running; then
      log "eval v2 daemon still running (pid=$(cat "$EVAL_V2_DAEMON_PID")); slept=${waited}s"
    elif [[ -f "$EVAL_V2_RESULTS/daemon.log" ]] && grep -q '^\[5/5\]' "$EVAL_V2_RESULTS/daemon.log" 2>/dev/null; then
      log "eval v2 step 5 started but metrics not ready yet; slept=${waited}s"
    else
      log "eval v2 daemon not running and metrics missing; slept=${waited}s"
    fi
    if [[ "$WAIT_TIMEOUT_SEC" -gt 0 && "$waited" -ge "$WAIT_TIMEOUT_SEC" ]]; then
      log "ERROR: timeout waiting for $EVAL_V2_METRICS"
      exit 4
    fi
    sleep "$WAIT_POLL_SEC"
    waited=$((waited + WAIT_POLL_SEC))
  done
  log "eval v2 pipeline complete: $EVAL_V2_METRICS"
}

if [[ ! -f "$EVAL_V2_DATASET" ]]; then
  log "ERROR: missing v2 eval dataset: $EVAL_V2_DATASET"
  exit 3
fi

ERROR_EVAL_SIZE="$(dataset_sample_count)"
if [[ "$ERROR_EVAL_SIZE" -lt 1 ]]; then
  log "ERROR: v2 eval dataset empty: $EVAL_V2_DATASET"
  exit 3
fi
log "v2 eval dataset samples=$ERROR_EVAL_SIZE path=$EVAL_V2_DATASET"

wait_for_eval_v2

export DATA_DIR
export RESULT_ROOT
export CATALOG_ROOT
export REPORT_OUTPUT
export STAMP_FILE
export ANNOTATED_ERROR_EVAL="$EVAL_V2_DATASET"
export ERROR_EVAL_SIZE
export ERROR_DATASET="$DATA_DIR/error_eval_dataset_100.json"
export LLM_BACKEND="${LLM_BACKEND:-local}"
export SKIP_SEMANTIC="${SKIP_SEMANTIC:-1}"
export FULL_CATALOG_BUILD="${FULL_CATALOG_BUILD:-1}"

log "starting scale curve on v2 eval set (local 30B)"
log "  DATA_DIR=$DATA_DIR"
log "  RESULT_ROOT=$RESULT_ROOT"
log "  ERROR_EVAL_SIZE=$ERROR_EVAL_SIZE"

exec bash "$ROOT/evaluation/experiments/run_scale_error_curve_local.sh"
