#!/usr/bin/env bash
# Waits until `scripts/run_baseline_and_precision_ablations.sh` finishes (presence of
# e2e_ablation_score4_<WAIT_STAMP>/question_metrics.json), then runs the 4B check-model
# main pipeline + baseline.
#
#   nohup bash scripts/chain_dualchain_batch_then_check4b.sh > results/_chain_batch_then_4b.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WAIT_STAMP="${WAIT_STAMP:-$(cat "$ROOT/results/_batch_baseline_ablations_stamp.txt")}"
MARKER="$ROOT/results/e2e_ablation_score4_${WAIT_STAMP}/question_metrics.json"

echo "[chain] WAIT_STAMP=$WAIT_STAMP"
echo "[chain] waiting for marker: $MARKER"
while [[ ! -f "$MARKER" ]]; do
  echo "[chain] $(date -u +%Y-%m-%dT%H:%M:%SZ) pending..."
  sleep 120
done
echo "[chain] marker found; starting 4b pipeline + baseline"
exec bash scripts/run_dualchain_check_4b_pipeline_and_baseline.sh
