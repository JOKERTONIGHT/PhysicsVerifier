#!/usr/bin/env bash
# Recall attribution + conditional relabel pipeline (plan implementation).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUT="${OUT:-results/recall_attribution_v1}"
DATASET="${DATASET:-data/derived/eval_v2_deepseek_v4_pro_seed20260508/error_eval_dataset_100.cleaned.json}"
EXISTING_AUDIT="${EXISTING_AUDIT:-data/derived/eval_v2_deepseek_v4_pro_seed20260508/annotation_reasonableness_audit.json}"

RULE_RESULTS="${RULE_RESULTS:-results/scale_curve_error_v2_local_30b/scale_1500_cleaned/error_verifier_results.json}"
TARGETED_RESULTS="${TARGETED_RESULTS:-results/scale_curve_error_v2_local_30b/ab_matrix_1500_cleaned/TargetedRules/error_verifier_results.json}"
LOCAL30_RESULTS="${LOCAL30_RESULTS:-results/semantic_pure_check_cleaned_1500/local_30b/checker_results.json}"
GEMINI_RESULTS="${GEMINI_RESULTS:-results/semantic_pure_check_cleaned_1500/gemini_flash/checker_results.json}"
LOCAL30_LOG="${LOCAL30_LOG:-results/semantic_pure_check_cleaned_1500/local_30b/checker.log}"
GEMINI_LOG="${GEMINI_LOG:-results/semantic_pure_check_cleaned_1500/gemini_flash/checker.log}"

RUN_EXHAUSTIVE="${RUN_EXHAUSTIVE:-1}"
EXHAUSTIVE_MAX_SAMPLES="${EXHAUSTIVE_MAX_SAMPLES:-20}"
LOCAL_MODEL="${LOCAL_MODEL:-qwen3-30b-a3b-instruct-2507}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8766}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-local-vllm}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

mkdir -p "$OUT"

log "Phase 1: recall cause diagnostics"
"$PYTHON" scripts/recall_cause_diagnostics.py \
  --dataset "$DATASET" \
  --output-dir "$OUT/phase1_forensics" \
  --experiment rules_baseline "$RULE_RESULTS" \
  --experiment targeted_rules "$TARGETED_RESULTS" \
  --experiment local_30b_checker "$LOCAL30_RESULTS" \
  --experiment gemini_flash_checker "$GEMINI_RESULTS"

log "Phase 2: matching sensitivity"
MATCH_OUT="$OUT/phase2_matching/matching_sensitivity_report.json"
mkdir -p "$OUT/phase2_matching"
"$PYTHON" - <<PY
import json
from pathlib import Path
import subprocess, sys
root = Path("$OUT/phase2_matching")
rows = []
for label, results in [
    ("rules_baseline", "$RULE_RESULTS"),
    ("targeted_rules", "$TARGETED_RESULTS"),
    ("local_30b_checker", "$LOCAL30_RESULTS"),
    ("gemini_flash", "$GEMINI_RESULTS"),
]:
    out = root / f"{label}.json"
    subprocess.check_call([
        sys.executable, "scripts/evaluate_match_sensitivity.py",
        "--dataset", "$DATASET",
        "--results", results,
        "--output", str(out),
        "--label", label,
    ])
    rows.append(json.loads(out.read_text(encoding="utf-8")))
report = {"dataset": "$DATASET", "experiments": rows}
(root / "matching_sensitivity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"experiments": len(rows)}, ensure_ascii=False))
PY

log "Phase 3: annotation forensics (mapped from existing audit)"
"$PYTHON" scripts/audit_annotation_forensics.py \
  --dataset "$DATASET" \
  --from-audit "$EXISTING_AUDIT" \
  --forensics "$OUT/phase1_forensics/common_missed_gt_report.json" \
  --fp-report "$OUT/phase1_forensics/checker_fp_or_missed_label_report.json" \
  --output "$OUT/phase3_annotation/annotation_forensics_report.json"

log "Phase 4: checker upper-bound ablation"
EXH="$OUT/phase4_checker/exhaustive_subset/checker_results.json"
mkdir -p "$OUT/phase4_checker/exhaustive_subset"
if [[ "$RUN_EXHAUSTIVE" == "1" ]]; then
  PRIORITY="$OUT/phase1_forensics/priority_sample_ids.json"
  "$PYTHON" - <<PY
import json
from pathlib import Path
common = json.loads(Path("$OUT/phase1_forensics/common_missed_gt_report.json").read_text(encoding="utf-8"))
ids = sorted({str(x.get("sample_id")) for x in (common.get("common_to_all_experiments") or []) if x.get("sample_id")})
if not ids:
    data = json.loads(Path("$DATASET").read_text(encoding="utf-8"))
    ids = [str(r.get("id")) for r in data[:$EXHAUSTIVE_MAX_SAMPLES]]
else:
    ids = ids[:$EXHAUSTIVE_MAX_SAMPLES]
Path("$PRIORITY").write_text(json.dumps(ids, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(ids))
PY
  export OPENAI_API_KEY="$OPENAI_API_KEY_LOCAL"
  export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
  export OPENAI_API_BASE="$OPENAI_BASE_URL"
  export OPENAI_DISABLE_THINKING="${OPENAI_DISABLE_THINKING:-1}"
  "$PYTHON" scripts/run_llm_checker_baseline.py \
    --input "$DATASET" \
    --model "$LOCAL_MODEL" \
    --mode exhaustive \
    --sample-ids-file "$OUT/phase1_forensics/priority_sample_ids.json" \
    --out_json "$EXH" \
    --timeout 180 \
    --no-tqdm \
    2>&1 | tee "$OUT/phase4_checker/exhaustive_subset/checker.log" || true
fi

"$PYTHON" scripts/run_checker_upper_bound_ablation.py \
  --dataset "$DATASET" \
  --single-results "$LOCAL30_RESULTS" \
  --exhaustive-results "$EXH" \
  --single-log "$LOCAL30_LOG" \
  --exhaustive-log "$OUT/phase4_checker/exhaustive_subset/checker.log" \
  --output "$OUT/phase4_checker/upper_bound_checker_ablation.json"

log "Phase 5: decision gate"
"$PYTHON" scripts/generate_decision_gate_report.py \
  --forensics "$OUT/phase1_forensics/recall_failure_forensics.json" \
  --annotation "$OUT/phase3_annotation/annotation_forensics_report.json" \
  --matching "$OUT/phase2_matching/matching_sensitivity_report.json" \
  --checker-ablation "$OUT/phase4_checker/upper_bound_checker_ablation.json" \
  --output-json "$OUT/phase5_decision/decision_gate_report.json" \
  --output-md "$OUT/phase5_decision/dataset_revision_decision.md"

log "Phase 6: conditional dataset revision"
REVISED="$OUT/phase6_revision/error_eval_dataset_100.revised.json"
"$PYTHON" scripts/apply_conditional_dataset_revision.py \
  --dataset "$DATASET" \
  --annotation-forensics "$OUT/phase3_annotation/annotation_forensics_report.json" \
  --decision "$OUT/phase5_decision/decision_gate_report.json" \
  --output "$REVISED" \
  --report "$OUT/phase6_revision/dataset_revision_report.json"

log "Phase 7: reliability and bound report"
"$PYTHON" scripts/generate_reliability_bound_report.py \
  --dataset "$DATASET" \
  --forensics "$OUT/phase1_forensics/recall_failure_forensics.json" \
  --annotation "$OUT/phase3_annotation/annotation_forensics_report.json" \
  --matching "$OUT/phase2_matching/matching_sensitivity_report.json" \
  --checker-ablation "$OUT/phase4_checker/upper_bound_checker_ablation.json" \
  --decision "$OUT/phase5_decision/decision_gate_report.json" \
  --revision-report "$OUT/phase6_revision/dataset_revision_report.json" \
  --output-dir "$OUT/final_report"

log "done -> $OUT"
