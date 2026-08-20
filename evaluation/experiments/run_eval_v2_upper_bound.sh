#!/usr/bin/env bash
set -euo pipefail

# Regenerate v2 eval set with a strong annotator, audit annotations, then estimate
# semantic recall upper bound using the same strong model as pure semantic checker.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
ANNOTATOR_MODEL="${ANNOTATOR_MODEL:-deepseek-v4-pro}"
UPPER_BOUND_MODEL="${UPPER_BOUND_MODEL:-deepseek-v4-pro}"
SEED="${SEED:-20260508}"
RECALL_SIZE="${RECALL_SIZE:-100}"
RUN_TAG="${RUN_TAG:-eval_v2_deepseek_v4_pro_seed${SEED}}"
OUTDIR="${OUTDIR:-data/derived/${RUN_TAG}}"
RESULTS_DIR="${RESULTS_DIR:-results/${RUN_TAG}}"
QA_INPUT="${QA_INPUT:-data/derived/combined_language_dual_chain_seed20260508_test200/qa_chain/combined_language_main_test.json}"
FIXED_IDS="${FIXED_IDS:-data/derived/combined_language_dual_chain_seed20260508_test200/fixed_recall_sample_ids_100.json}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
END_STEP="${END_STEP:-5}"

mkdir -p "$OUTDIR" "$RESULTS_DIR"

START_STEP="${START_STEP:-1}"
if [[ "$START_STEP" -le 1 && "$END_STEP" -ge 1 ]]; then
echo "[1/5] Build error-level eval set with ${ANNOTATOR_MODEL}"
BUILD_ARGS=(
  --input "$QA_INPUT"
  --recall-input "$QA_INPUT"
  --precision-input "$QA_INPUT"
  --error-output "$OUTDIR/error_eval_dataset_${RECALL_SIZE}.json"
  --question-output "$OUTDIR/question_eval_dataset_50_50.json"
  --precision-output "$OUTDIR/question_right_only_50.json"
  --recall-size "$RECALL_SIZE"
  --question-recall-size 50
  --precision-size 50
  --seed "$SEED"
  --strong-model "$ANNOTATOR_MODEL"
  --max-errors 0
  --fixed-sample-ids "$FIXED_IDS"
)
if [[ -n "${RESUME_DATASET:-}" ]]; then
  BUILD_ARGS+=(--resume-dataset "$RESUME_DATASET")
fi
if [[ "${SKIP_PRECISION_REBUILD:-0}" == "1" ]]; then
  BUILD_ARGS+=(--skip-precision)
fi
if [[ "$MAX_SAMPLES" =~ ^[1-9][0-9]*$ ]]; then
  BUILD_ARGS+=(--recall-size "$MAX_SAMPLES")
fi
"$PYTHON" scripts/build_physics_eval_sets.py "${BUILD_ARGS[@]}" 2>&1 | tee -a "$RESULTS_DIR/build_annotated_chain.log"
fi

if [[ "$START_STEP" -le 2 && "$END_STEP" -ge 2 ]]; then
echo "[2/5] Structural annotation audit (report-only)"
"$PYTHON" scripts/audit_eval_set_quality.py \
  --recall-dataset "$OUTDIR/error_eval_dataset_${RECALL_SIZE}.json" \
  --output "$OUTDIR/error_quality_audit.json" \
  2>&1 | tee "$RESULTS_DIR/error_quality_audit.log"
fi

if [[ "$START_STEP" -le 3 && "$END_STEP" -ge 3 ]]; then
echo "[3/5] Semantic annotation reasonableness audit with ${ANNOTATOR_MODEL}"
AUDIT_SAMPLES="${AUDIT_SAMPLES:-100}"
"$PYTHON" scripts/audit_annotation_reasonableness.py \
  --dataset "$OUTDIR/error_eval_dataset_${RECALL_SIZE}.json" \
  --output "$OUTDIR/annotation_reasonableness_audit.json" \
  --model "$ANNOTATOR_MODEL" \
  --max-samples "$AUDIT_SAMPLES" \
  2>&1 | tee "$RESULTS_DIR/annotation_reasonableness_audit.log"
fi

if [[ "$START_STEP" -le 4 && "$END_STEP" -ge 4 ]]; then
echo "[4/5] Pure semantic checker upper bound with ${UPPER_BOUND_MODEL}"
"$PYTHON" scripts/run_llm_checker_baseline.py \
  --input "$OUTDIR/error_eval_dataset_${RECALL_SIZE}.json" \
  --model "$UPPER_BOUND_MODEL" \
  --out_json "$RESULTS_DIR/upper_bound_checker_results.json" \
  --flush-every 1 \
  --progress-interval 5 \
  2>&1 | tee "$RESULTS_DIR/upper_bound_checker.log"
fi

if [[ "$START_STEP" -le 5 && "$END_STEP" -ge 5 ]]; then
echo "[5/5] Semantic recall upper bound evaluation"
"$PYTHON" scripts/evaluate_physics_eval_sets.py \
  --dataset "$OUTDIR/error_eval_dataset_${RECALL_SIZE}.json" \
  --results "$RESULTS_DIR/upper_bound_checker_results.json" \
  --audit "$RESULTS_DIR/upper_bound_checker_results.json" \
  --output "$RESULTS_DIR/upper_bound_error_metrics.json" \
  --match-mode semantic \
  --semantic-match-model "$UPPER_BOUND_MODEL" \
  2>&1 | tee "$RESULTS_DIR/upper_bound_error_metrics.log"
fi

echo "Done."
echo "Dataset: $OUTDIR/error_eval_dataset_${RECALL_SIZE}.json"
echo "Upper bound metrics: $RESULTS_DIR/upper_bound_error_metrics.json"
