#!/usr/bin/env bash
# A/B matrix for scale_1500_cleaned optimization plan.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-data/derived/expansion3000_scale_v2_eval_deepseek_seed20260508}"
ERROR_DATASET="${ERROR_DATASET:-data/derived/eval_v2_deepseek_v4_pro_seed20260508/error_eval_dataset_100.cleaned.json}"
RESULT_ROOT="${RESULT_ROOT:-results/scale_curve_error_v2_local_30b}"
CATALOG_ROOT="${CATALOG_ROOT:-catalogs/scale_curve_error_v2_local_30b}"
BASE_CATALOG="${BASE_CATALOG:-$CATALOG_ROOT/rules_unified_scale_1500.json}"
TARGETED_CATALOG="${TARGETED_CATALOG:-$CATALOG_ROOT/rules_unified_scale_1500_targeted.json}"
MODEL="${MODEL:-qwen3-30b-a3b-instruct-2507}"
VLLM_SCRIPT="${VLLM_SCRIPT:-/home/jinjianhan/deploy/qwen3_30b/start_vllm_30b.sh}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8766}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-32768}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-local-vllm}"
MANIFEST="${EXPERIENCE_CODE_MANIFEST:-results/experience_symbolic_program_manifest_v2_unified.json}"
MODULE="${EXPERIENCE_CODE_MODULE:-symbolic.generated_experience_checks_v2_unified}"
MATRIX_ROOT="${MATRIX_ROOT:-$RESULT_ROOT/ab_matrix_1500_cleaned}"
RUN_VARIANTS="${RUN_VARIANTS:-all}"
SKIP_VERIFIER="${SKIP_VERIFIER:-0}"
SKIP_SEMANTIC="${SKIP_SEMANTIC:-0}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

wait_for_vllm() {
  local url="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
  local i=0
  while ! curl -sf "$url" >/dev/null 2>&1; do
    i=$((i + 1))
    if [[ "$i" -gt 120 ]]; then
      log "ERROR: vLLM not ready at $url"
      exit 4
    fi
    sleep 10
  done
  log "vLLM ready: $url"
}

start_vllm_if_needed() {
  if curl -sf "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    log "vLLM already running"
    return 0
  fi
  log "starting local 30B vLLM"
  MAX_LEN="$VLLM_MAX_LEN" bash "$VLLM_SCRIPT"
  wait_for_vllm
}

use_local_llm() {
  export OPENAI_API_KEY="$OPENAI_API_KEY_LOCAL"
  export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
  export OPENAI_API_BASE="$OPENAI_BASE_URL"
  export PHYSICSVERIFIER_LLM_CONTEXT_TOKENS="${PHYSICSVERIFIER_LLM_CONTEXT_TOKENS:-32768}"
}

should_run() {
  local name="$1"
  [[ "$RUN_VARIANTS" == "all" || "$RUN_VARIANTS" == *"$name"* ]]
}

prepare_targeted_catalog() {
  local baseline_by_rule="$MATRIX_ROOT/baseline/failure_analysis_by_rule.json"
  if [[ ! -f "$baseline_by_rule" ]]; then
    baseline_by_rule="$RESULT_ROOT/scale_1500_cleaned/failure_analysis_by_rule.json"
  fi
  if [[ ! -f "$baseline_by_rule" ]]; then
    log "WARN: missing failure_analysis_by_rule; targeted catalog will only add theme rules"
    baseline_by_rule="$MATRIX_ROOT/baseline/failure_analysis_by_rule.json"
    mkdir -p "$(dirname "$baseline_by_rule")"
    echo '{"top_fp_rules":[],"top_missed_gt_themes":[]}' > "$baseline_by_rule"
  fi
  "$PYTHON" scripts/apply_targeted_rule_patches_1500.py \
    --base-catalog "$BASE_CATALOG" \
    --failure-by-rule "$baseline_by_rule" \
    --output "$TARGETED_CATALOG"
}

run_variant() {
  local label="$1"
  shift
  local out_dir="${OUT_DIR_OVERRIDE:-$MATRIX_ROOT}/$label"
  mkdir -p "$out_dir"
  cp -f "$ERROR_DATASET" "$out_dir/error_eval_dataset_100.json"

  if [[ "$SKIP_VERIFIER" != "1" ]]; then
    start_vllm_if_needed
    use_local_llm
    local catalog="$BASE_CATALOG"
    local extra_flags=("$@")
    if [[ "$label" == "TargetedRules" ]]; then
      catalog="$TARGETED_CATALOG"
    fi
    log "variant=$label catalog=$catalog"
    "$PYTHON" scripts/run_verifier.py \
      --input "$out_dir/error_eval_dataset_100.json" \
      --output "$out_dir/error_verifier_results.json" \
      --symbolic-output "$out_dir/error_symbolic_audit.json" \
      --full-output "$out_dir/error_verifier_full.json" \
      --model "$MODEL" \
      --unified-catalog "$catalog" \
      --experience-code-manifest "$MANIFEST" \
      --experience-code-module "$MODULE" \
      --progress-interval 10 \
      --no-symbolic-check \
      "${extra_flags[@]}" \
      2>&1 | tee "$out_dir/run_verifier.log"
  fi

  local eval_args=(
    --dataset "$out_dir/error_eval_dataset_100.json"
    --results "$out_dir/error_verifier_results.json"
    --audit "$out_dir/error_symbolic_audit.json"
    --output-dir "$out_dir"
    --label "$label"
  )
  if [[ "$SKIP_SEMANTIC" == "1" ]]; then
    eval_args+=(--skip-semantic)
  fi
  "$PYTHON" scripts/evaluate_scale_with_semantic.py "${eval_args[@]}"
}

run_strict_gate_sweep() {
  local sweep_dir="$MATRIX_ROOT/strict_gate_sweep"
  mkdir -p "$sweep_dir"
  OUT_DIR_OVERRIDE="$sweep_dir" run_variant "StrictGate_score5_quote05_para1_sample6_top4" \
    --min-diagnostic-rule-score 5.0 \
    --quote-symbol-ratio 0.5 \
    --max-per-paragraph 1 \
    --max-per-sample 6 \
    --unified-rule-top-n 4
  OUT_DIR_OVERRIDE="$sweep_dir" run_variant "StrictGate_score5_quote05_para1_sample6" \
    --min-diagnostic-rule-score 5.0 \
    --quote-symbol-ratio 0.5 \
    --max-per-paragraph 1 \
    --max-per-sample 6
  OUT_DIR_OVERRIDE="$sweep_dir" run_variant "StrictGate_score6_quote05_para1_sample6" \
    --min-diagnostic-rule-score 6.0 \
    --quote-symbol-ratio 0.5 \
    --max-per-paragraph 1 \
    --max-per-sample 6
  OUT_DIR_OVERRIDE="$sweep_dir" run_variant "StrictGate_score5_para1_sample6" \
    --min-diagnostic-rule-score 5.0 \
    --max-per-paragraph 1 \
    --max-per-sample 6
  "$PYTHON" - <<'PY' "$sweep_dir"
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for d in sorted(root.glob("StrictGate_*")):
    p = d / "dual_eval_summary.json"
    if not p.exists():
        continue
    obj = json.loads(p.read_text(encoding="utf-8"))
    loc = obj.get("location") or {}
    rows.append({
        "label": obj.get("label"),
        "recall": loc.get("recall"),
        "precision": loc.get("precision"),
        "f1": loc.get("f1"),
        "semantic_recall": (obj.get("semantic") or {}).get("recall"),
    })
rows.sort(key=lambda x: (-(x.get("precision") or 0), x.get("f1") or 0))
out = root / "strict_gate_sweep_summary.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(out)
PY
}

preflight() {
  for f in "$ERROR_DATASET" "$BASE_CATALOG"; do
    [[ -f "$f" ]] || { log "ERROR: missing $f"; exit 2; }
  done
  mkdir -p "$MATRIX_ROOT"
}

main() {
  preflight
  if should_run baseline; then
    run_variant baseline
  fi
  if should_run StrictGate; then
    run_strict_gate_sweep
  fi
  if should_run ValidatorOnly; then
    run_variant ValidatorOnly --enable-diagnostic-validator
  fi
  if should_run MergeOnly; then
    run_variant MergeOnly --enable-diagnostic-aggregator
  fi
  if should_run ValidatorMerge; then
    run_variant ValidatorMerge --enable-diagnostic-validator --enable-diagnostic-aggregator
  fi
  if should_run TargetedRules; then
    prepare_targeted_catalog
    run_variant TargetedRules --enable-diagnostic-validator --enable-diagnostic-aggregator
  fi
  log "A/B matrix complete -> $MATRIX_ROOT"
}

main "$@"
