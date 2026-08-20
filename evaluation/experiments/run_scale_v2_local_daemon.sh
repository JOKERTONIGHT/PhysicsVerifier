#!/usr/bin/env bash
# Detached runner: wait for eval v2 completion, then scale-curve expansion with local 30B.
#
# Usage:
#   bash scripts/run_scale_v2_local_daemon.sh start
#   bash scripts/run_scale_v2_local_daemon.sh status
#   bash scripts/run_scale_v2_local_daemon.sh stop
#
# Env (optional):
#   EVAL_V2_TAG=eval_v2_deepseek_v4_pro_seed20260508
#   DATA_DIR=data/derived/expansion3000_scale_v2_eval_deepseek_seed20260508
#   RESULT_ROOT=results/scale_curve_error_v2_local_30b
#   LLM_BACKEND=local
#   SCALES=300,600,900,1200,1500,1800,2100,2400,2700

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

RUN_TAG="${RUN_TAG:-scale_v2_local_30b}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_TAG}}"
PID_FILE="$RESULTS_DIR/daemon.pid"
LOG_FILE="$RESULTS_DIR/daemon.log"
WRAPPER="$ROOT_DIR/scripts/run_scale_v2_local_after_eval.sh"

mkdir -p "$RESULTS_DIR"

is_running() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

cmd_status() {
  if is_running; then
    echo "running pid=$(cat "$PID_FILE")"
    echo "log=$LOG_FILE"
    grep -E 'waiting for eval v2|eval v2 pipeline complete|starting scale curve|scale=' "$LOG_FILE" 2>/dev/null | tail -5 || true
    if [[ -f "${RESULT_ROOT:-results/scale_curve_error_v2_local_30b}/curve_metrics.csv" ]]; then
      echo "curve_metrics=$(readlink -f "${RESULT_ROOT:-results/scale_curve_error_v2_local_30b}/curve_metrics.csv" 2>/dev/null || echo yes)"
    fi
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

  echo "starting detached scale-v2 pipeline (waits for eval v2, then local 30B scale curve)"
  echo "log=$LOG_FILE"
  echo "env: EVAL_V2_TAG=${EVAL_V2_TAG:-eval_v2_deepseek_v4_pro_seed20260508} RESULT_ROOT=${RESULT_ROOT:-results/scale_curve_error_v2_local_30b} LLM_BACKEND=${LLM_BACKEND:-local}"

  nohup setsid bash -c "
    cd \"$ROOT_DIR\"
    export EVAL_V2_TAG=\"${EVAL_V2_TAG:-eval_v2_deepseek_v4_pro_seed20260508}\"
    export DATA_DIR=\"${DATA_DIR:-data/derived/expansion3000_scale_v2_eval_deepseek_seed20260508}\"
    export RESULT_ROOT=\"${RESULT_ROOT:-results/scale_curve_error_v2_local_30b}\"
    export CATALOG_ROOT=\"${CATALOG_ROOT:-catalogs/scale_curve_error_v2_local_30b}\"
    export REPORT_OUTPUT=\"${REPORT_OUTPUT:-docs/规则库规模曲线实验报告_v2_local_30b.md}\"
    export LLM_BACKEND=\"${LLM_BACKEND:-local}\"
    export SCALES=\"${SCALES:-300,600,900,1200,1500,1800,2100,2400,2700}\"
    export SKIP_SEMANTIC=\"${SKIP_SEMANTIC:-1}\"
    export FULL_CATALOG_BUILD=\"${FULL_CATALOG_BUILD:-1}\"
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
