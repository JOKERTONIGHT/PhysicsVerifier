#!/usr/bin/env bash
# Scale-curve experiment via remote OpenAI-compatible API (yeysai / .env).
#
# Usage (detach-safe):
#   cd /home/jinjianhan/PhysicsVerifier
#   nohup bash scripts/run_scale_error_curve_api.sh \
#     > results/_scale_error_curve_v3_api_batch.log 2>&1 &
#   echo $! > results/_scale_error_curve_v3_api_batch.pid
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export LLM_BACKEND=api
export RESULT_ROOT="${RESULT_ROOT:-results/scale_curve_error_v3_api}"
export CATALOG_ROOT="${CATALOG_ROOT:-catalogs/scale_curve_error_v3_api}"
export REPORT_OUTPUT="${REPORT_OUTPUT:-docs/规则库规模曲线实验报告_v3_api.md}"
export STAMP_FILE="${STAMP_FILE:-$ROOT/results/_scale_error_curve_v3_api_stamp.txt}"
export PHYSICSVERIFIER_LLM_CONTEXT_TOKENS="${PHYSICSVERIFIER_LLM_CONTEXT_TOKENS:-32768}"

# Avoid inheriting local vLLM overrides from the shell.
unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_API_BASE 2>/dev/null || true

exec bash "$ROOT/evaluation/experiments/run_scale_error_curve_local.sh" "$@"
