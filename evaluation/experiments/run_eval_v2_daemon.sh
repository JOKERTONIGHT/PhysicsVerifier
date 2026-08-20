#!/usr/bin/env bash
# Run eval v2 pipeline detached from terminal (survives SSH disconnect).
#
# Usage:
#   # Resume build only (recommended while annotating):
#   RESUME_DATASET=data/derived/eval_v2_deepseek_v4_pro_seed20260508/error_eval_dataset_100.json \
#   END_STEP=1 SKIP_PRECISION_REBUILD=1 bash scripts/run_eval_v2_daemon.sh
#
#   # After 100 samples annotated, run audit + upper bound:
#   START_STEP=2 END_STEP=5 bash scripts/run_eval_v2_daemon.sh
#
#   # Check status:
#   bash scripts/run_eval_v2_daemon.sh status
#
#   # Stop:
#   bash scripts/run_eval_v2_daemon.sh stop

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-eval_v2_deepseek_v4_pro_seed20260508}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_TAG}}"
PID_FILE="$RESULTS_DIR/daemon.pid"
LOG_FILE="$RESULTS_DIR/daemon.log"
WRAPPER="$ROOT_DIR/scripts/run_eval_v2_upper_bound.sh"

mkdir -p "$RESULTS_DIR"

is_running() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  if kill -0 "$pid" 2>/dev/null; then
    return 0
  fi
  return 1
}

cmd_status() {
  if is_running; then
    echo "running pid=$(cat "$PID_FILE")"
    echo "log=$LOG_FILE"
    grep '"progress"' "$LOG_FILE" 2>/dev/null | tail -3 || true
    grep '"resume_loaded_samples"' "$LOG_FILE" 2>/dev/null | tail -1 || true
  else
    echo "not running"
    if [[ -f "$PID_FILE" ]]; then
      echo "stale pid file: $PID_FILE"
    fi
  fi
}

cmd_stop() {
  if ! is_running; then
    echo "not running"
    rm -f "$PID_FILE"
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  echo "stopping pid=$pid"
  kill "$pid" 2>/dev/null || true
  sleep 2
  if kill -0 "$pid" 2>/dev/null; then
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "stopped"
}

cmd_start() {
  if is_running; then
    echo "already running pid=$(cat "$PID_FILE")"
    echo "log=$LOG_FILE"
    exit 0
  fi

  echo "starting detached pipeline"
  echo "log=$LOG_FILE"
  echo "env: RESUME_DATASET=${RESUME_DATASET:-} START_STEP=${START_STEP:-1} END_STEP=${END_STEP:-5} SKIP_PRECISION_REBUILD=${SKIP_PRECISION_REBUILD:-0}"

  # setsid + nohup: survive SSH disconnect and hangup signals.
  nohup setsid bash -c "
    cd \"$ROOT_DIR\"
    export RUN_TAG=\"${RUN_TAG}\"
    export RESUME_DATASET=\"${RESUME_DATASET:-}\"
    export START_STEP=\"${START_STEP:-1}\"
    export END_STEP=\"${END_STEP:-5}\"
    export SKIP_PRECISION_REBUILD=\"${SKIP_PRECISION_REBUILD:-0}\"
    export ANNOTATOR_MODEL=\"${ANNOTATOR_MODEL:-deepseek-v4-pro}\"
    export UPPER_BOUND_MODEL=\"${UPPER_BOUND_MODEL:-deepseek-v4-pro}\"
    exec bash \"$WRAPPER\"
  " >>"$LOG_FILE" 2>&1 &

  echo $! >"$PID_FILE"
  disown -h $! 2>/dev/null || true
  echo "started pid=$(cat "$PID_FILE")"
}

ACTION="${1:-start}"
case "$ACTION" in
  start) cmd_start ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  restart)
    cmd_stop
    cmd_start
    ;;
  *)
    echo "usage: $0 {start|status|stop|restart}" >&2
    exit 2
    ;;
esac
