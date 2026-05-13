#!/usr/bin/env bash
# Wait for dual-chain batch A (baseline+ablations) and optional 4B, then refresh docs.
# Idempotent: safe to run once; re-run after 4B if 4B was not ready on first pass.
#
#   nohup bash scripts/watch_dualchain_and_update_docs.sh >> results/_dualchain_doc_watcher.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(cat "$ROOT/results/_batch_baseline_ablations_stamp.txt" 2>/dev/null || true)"
if [[ -z "$STAMP" ]]; then
  echo "[watcher] error: missing results/_batch_baseline_ablations_stamp.txt" >&2
  exit 2
fi
MARK_A="$ROOT/results/e2e_ablation_score4_${STAMP}/question_metrics.json"
TRACK="$ROOT/docs/dual_chain_experiment_tracking_20260510.md"

echo "[watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) STAMP_BATCH=$STAMP waiting for $MARK_A"
until [[ -f "$MARK_A" ]]; do
  sleep 120
  echo "[watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) still waiting batch A..."
done
echo "[watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) batch A metrics present"

STAMP_BATCH="$STAMP" "$PYTHON" "$ROOT/scripts/update_experiment_plan_dualchain.py" --phase batch
{
  echo ""
  echo "---"
  echo "<!-- watcher batch A $(date -u +%Y-%m-%dT%H:%M:%SZ) STAMP=$STAMP -->"
  STAMP_BATCH="$STAMP" "$PYTHON" "$ROOT/scripts/emit_dual_chain_results_md.py"
} >> "$TRACK"

echo "[watcher] waiting for 4B stamp + baseline_check_4b question_metrics (optional long wait)..."
until [[ -f "$ROOT/results/_dualchain_check4b_stamp.txt" ]]; do
  sleep 60
  echo "[watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) no _dualchain_check4b_stamp.txt yet"
done
STAMP4="$(cat "$ROOT/results/_dualchain_check4b_stamp.txt")"
MARK_B="$ROOT/results/baseline_check_4b_${STAMP4}/question_metrics.json"
until [[ -f "$MARK_B" ]]; do
  sleep 120
  echo "[watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) waiting 4B baseline metrics STAMP_4B=$STAMP4"
done
echo "[watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) 4B done"

STAMP_BATCH="$STAMP" STAMP_4B="$STAMP4" "$PYTHON" "$ROOT/scripts/update_experiment_plan_dualchain.py" --phase fourb
{
  echo ""
  echo "---"
  echo "<!-- watcher 4B $(date -u +%Y-%m-%dT%H:%M:%SZ) STAMP_4B=$STAMP4 -->"
  STAMP_BATCH="$STAMP" STAMP_4B="$STAMP4" "$PYTHON" "$ROOT/scripts/emit_dual_chain_results_md.py"
} >> "$TRACK"

echo "[watcher] $(date -u +%Y-%m-%dT%H:%M:%SZ) all doc updates finished"
