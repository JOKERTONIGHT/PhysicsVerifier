#!/usr/bin/env bash
# Sequential daemon for scale_1500 A/B jobs (strict gate then targeted rules).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
LOG_DIR="${LOG_DIR:-results/scale_curve_error_v2_local_30b/ab_matrix_1500_cleaned/logs}"
mkdir -p "$LOG_DIR"
export RUN_VARIANTS="${RUN_VARIANTS:-StrictGate}"
export SKIP_SEMANTIC=1
bash scripts/run_scale_1500_ab_matrix.sh 2>&1 | tee "$LOG_DIR/sequential_ab.log"
export RUN_VARIANTS=TargetedRules
bash scripts/run_scale_1500_ab_matrix.sh 2>&1 | tee -a "$LOG_DIR/sequential_ab.log"
