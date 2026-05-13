#!/usr/bin/env bash
# Dual-chain eval (100 error + 100 question) with **check model** qwen3-4b-instruct-2507:
#   A) Full PhysicsVerifier pipeline (default rule pool / score gates from code + pipeline CLI)
#   B) Semantic-only baseline with the same check model
#
# GT 标注与测评集与 30B 检查模型实验相同（skip-build 仅跑 verifier / baseline）。
#
# Usage:
#   CHECK_MODEL=qwen3-4b-instruct-2507 bash scripts/run_dualchain_check_4b_pipeline_and_baseline.sh
#   nohup bash scripts/run_dualchain_check_4b_pipeline_and_baseline.sh > results/_dualchain_4b_nohup.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "[error] PYTHON not executable: $PYTHON" >&2
  exit 2
fi

DATASET_DIR="${DATASET_DIR:-data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain}"
ERROR_DATASET="${ERROR_DATASET:-$DATASET_DIR/error_eval_dataset_100.json}"
QUESTION_DATASET="${QUESTION_DATASET:-$DATASET_DIR/question_eval_dataset_50_50.json}"
PRECISION_DATASET="${PRECISION_DATASET:-$DATASET_DIR/question_right_only_50.json}"
CHECK_MODEL="${CHECK_MODEL:-qwen3-4b-instruct-2507}"
STRONG_MODEL="${STRONG_MODEL:-qwen3-30b-a3b-instruct-2507}"
UNIFIED_CATALOG="${UNIFIED_CATALOG:-catalogs/unified_rule_library_v2_llm_enhanced_20260504.json}"
MANIFEST="${EXPERIENCE_CODE_MANIFEST:-results/experience_symbolic_program_manifest_v2_unified.json}"
MODULE="${EXPERIENCE_CODE_MODULE:-symbolic.generated_experience_checks_v2_unified}"
SYMBOLIC_TOPIC_CHECK_LIMIT="${SYMBOLIC_TOPIC_CHECK_LIMIT:-32}"

EMPTY_AUDIT="${EMPTY_AUDIT:-$ROOT/results/_empty_symbolic_audit.json}"
[[ -s "$EMPTY_AUDIT" ]] || echo '[]' > "$EMPTY_AUDIT"

RECALL_SIZE="${RECALL_SIZE:-100}"
PRECISION_SIZE="${PRECISION_SIZE:-50}"
STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
echo "$STAMP" > "$ROOT/results/_dualchain_check4b_stamp.txt"

TAG_MAIN="${CHECK4B_MAIN_TAG:-e2e_main_check_4b}"
TAG_BASE="${CHECK4B_BASE_TAG:-baseline_check_4b}"

MAIN_OUT="$ROOT/results/${TAG_MAIN}_${STAMP}"
BASE_OUT="$ROOT/results/${TAG_BASE}_${STAMP}"

mkdir -p "$MAIN_OUT" "$BASE_OUT"
echo "$CHECK_MODEL" > "$MAIN_OUT/check_model.txt"
echo "$CHECK_MODEL" > "$BASE_OUT/model.txt"

echo "[4b-main] STAMP=$STAMP -> $MAIN_OUT"
cp -f "$ERROR_DATASET" "$MAIN_OUT/error_eval_dataset_${RECALL_SIZE}.json"
cp -f "$QUESTION_DATASET" "$MAIN_OUT/question_eval_dataset_${RECALL_SIZE}_${PRECISION_SIZE}.json"
if [[ -f "$PRECISION_DATASET" ]]; then
  cp -f "$PRECISION_DATASET" "$MAIN_OUT/question_right_only_${PRECISION_SIZE}.json"
fi

"$PYTHON" scripts/run_physics_eval_pipeline.py \
  --python "$PYTHON" \
  --output-dir "$MAIN_OUT" \
  --recall-size "$RECALL_SIZE" \
  --precision-size "$PRECISION_SIZE" \
  --strong-model "$STRONG_MODEL" \
  --check-model "$CHECK_MODEL" \
  --unified-catalog "$UNIFIED_CATALOG" \
  --experience-code-manifest "$MANIFEST" \
  --experience-code-module "$MODULE" \
  --symbolic-topic-check-limit "$SYMBOLIC_TOPIC_CHECK_LIMIT" \
  --max-per-sample "${MAX_PER_SAMPLE:-12}" \
  --max-per-paragraph "${MAX_PER_PARAGRAPH:-2}" \
  --skip-build \
  2>&1 | tee "$MAIN_OUT/pipeline.log"

echo "[4b-baseline] -> $BASE_OUT"
{
  echo "=== Semantic baseline ($CHECK_MODEL) start ==="
  "$PYTHON" scripts/run_llm_checker_baseline.py \
    --input "$ERROR_DATASET" --model "$CHECK_MODEL" \
    --out_json "$BASE_OUT/error_verifier_results.json"
  "$PYTHON" scripts/run_llm_checker_baseline.py \
    --input "$QUESTION_DATASET" --model "$CHECK_MODEL" \
    --out_json "$BASE_OUT/question_verifier_results.json"
  "$PYTHON" scripts/evaluate_physics_eval_sets.py \
    --dataset "$ERROR_DATASET" \
    --results "$BASE_OUT/error_verifier_results.json" \
    --audit "$EMPTY_AUDIT" \
    --output "$BASE_OUT/error_metrics.json" \
    --match-mode location
  "$PYTHON" scripts/evaluate_question_level_sets.py \
    --dataset "$QUESTION_DATASET" \
    --results "$BASE_OUT/question_verifier_results.json" \
    --audit "$EMPTY_AUDIT" \
    --output "$BASE_OUT/question_metrics.json"
  echo "=== Semantic baseline ($CHECK_MODEL) done ==="
} | tee "$BASE_OUT/run.log"

echo "[ok] 4b check-model batch STAMP=$STAMP $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "  main pipeline: $MAIN_OUT"
echo "  baseline:      $BASE_OUT"

SB="${STAMP_BATCH:-$(cat "$ROOT/results/_batch_baseline_ablations_stamp.txt" 2>/dev/null || echo "")}"
if [[ -n "$SB" ]]; then
  {
    echo ""
    echo "---"
    echo "<!-- auto tables $(date -u +%Y-%m-%dT%H:%M:%SZ) batch=$SB check4b=$STAMP -->"
    STAMP_BATCH="$SB" STAMP_4B="$STAMP" "$PYTHON" "$ROOT/scripts/emit_dual_chain_results_md.py"
  } >> "$ROOT/docs/dual_chain_experiment_tracking_20260510.md" || true
fi
