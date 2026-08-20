#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATASET="data/derived/eval_v2_deepseek_v4_pro_seed20260508/error_eval_dataset_100.json"
AUDIT_OUT="data/derived/eval_v2_deepseek_v4_pro_seed20260508/annotation_reasonableness_audit.json"
AUDIT_MERGED="data/derived/eval_v2_deepseek_v4_pro_seed20260508/annotation_reasonableness_audit.merged.json"
CLEANED="data/derived/eval_v2_deepseek_v4_pro_seed20260508/error_eval_dataset_100.cleaned.json"
RETRY_IDS="data/derived/eval_v2_deepseek_v4_pro_seed20260508/audit_retry_sample_ids.json"
LOG_DIR="results/scale_v2_1500_cleaned"
mkdir -p "$LOG_DIR"
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$LOG_DIR/pipeline.log"; }

log "[1/3] rerun audit for failed samples"
"$PYTHON" scripts/audit_annotation_reasonableness.py \
  --dataset "$DATASET" \
  --output "$AUDIT_MERGED" \
  --model deepseek-v4-pro \
  --sample-ids "$RETRY_IDS" \
  --merge-from "$AUDIT_OUT" \
  --batch-gt-size 4 \
  --max-samples 0 \
  2>&1 | tee -a "$LOG_DIR/audit_retry.log"
cp -f "$AUDIT_MERGED" "$AUDIT_OUT"

log "[2/3] cleanup dataset from audit labels"
"$PYTHON" scripts/apply_annotation_audit_cleanup.py \
  --dataset "$DATASET" \
  --audit "$AUDIT_OUT" \
  --output "$CLEANED" \
  --backup "${DATASET%.json}.before_cleanup.json" \
  --report "data/derived/eval_v2_deepseek_v4_pro_seed20260508/error_eval_cleanup_report.json"

log "[3/3] start scale 1500 cleaned eval daemon"
bash scripts/run_scale_1500_v2_cleaned_daemon.sh start
log "pipeline handoff complete"
