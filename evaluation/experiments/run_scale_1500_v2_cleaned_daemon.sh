#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RESULTS_DIR="${RESULTS_DIR:-results/scale_v2_1500_cleaned}"
PID_FILE="$RESULTS_DIR/daemon.pid"
LOG_FILE="$RESULTS_DIR/daemon.log"
WRAPPER="$ROOT_DIR/scripts/run_scale_1500_v2_cleaned.sh"

mkdir -p "$RESULTS_DIR"

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

cmd_status() {
  if is_running; then
    echo "running pid=$(cat "$PID_FILE")"
    echo "log=$LOG_FILE"
    tail -5 "$LOG_FILE" 2>/dev/null || true
  else
    echo "not running"
    [[ -f "$PID_FILE" ]] && echo "stale pid file: $PID_FILE"
  fi
}

cmd_stop() {
  if ! is_running; then
    rm -f "$PID_FILE"
    echo "not running"
    return 0
  fi
  kill "$(cat "$PID_FILE")" 2>/dev/null || true
  sleep 2
  rm -f "$PID_FILE"
  echo "stopped"
}

cmd_start() {
  if is_running; then
    echo "already running pid=$(cat "$PID_FILE")"
    exit 0
  fi
  echo "starting scale 1500 cleaned eval"
  echo "log=$LOG_FILE"
  nohup setsid bash -c "
    cd \"$ROOT_DIR\"
    export ERROR_DATASET=\"${ERROR_DATASET:-data/derived/eval_v2_deepseek_v4_pro_seed20260508/error_eval_dataset_100.cleaned.json}\"
    export RESULT_ROOT=\"${RESULT_ROOT:-results/scale_curve_error_v2_local_30b}\"
    export OUT_TAG=\"${OUT_TAG:-scale_1500_cleaned}\"
    exec bash \"$WRAPPER\"
  " >>"$LOG_FILE" 2>&1 &
  echo $! >"$PID_FILE"
  echo "started pid=$(cat "$PID_FILE")"
}

case "${1:-start}" in
  start) cmd_start ;;
  status) cmd_status ;;
  stop) cmd_stop ;;
  *) echo "usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac
