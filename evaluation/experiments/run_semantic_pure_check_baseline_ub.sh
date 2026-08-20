#!/usr/bin/env bash
# Pure LLM checker baseline (local 30B) + upper bound (Gemini 3 Flash API)
# on the current cleaned error-level eval set.
# Evaluation uses location matching (same settings as scale_1500_cleaned).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATASET="${DATASET:-data/derived/eval_v2_deepseek_v4_pro_seed20260508/error_eval_dataset_100.cleaned.json}"
OUT_ROOT="${OUT_ROOT:-results/semantic_pure_check_cleaned_1500}"
LOCAL_MODEL="${LOCAL_MODEL:-qwen3-30b-a3b-instruct-2507}"
UPPER_MODEL="${UPPER_MODEL:-gemini-3-flash-preview}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8766}"
VLLM_SCRIPT="${VLLM_SCRIPT:-/home/jinjianhan/deploy/qwen3_30b/start_vllm_30b.sh}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-local-vllm}"
RUN_LOCAL="${RUN_LOCAL:-1}"
RUN_UPPER="${RUN_UPPER:-1}"
PHASE="${PHASE:-all}"
LOCATION_IOU_THRESHOLD="${LOCATION_IOU_THRESHOLD:-0.5}"
LOCATION_COVERAGE_THRESHOLD="${LOCATION_COVERAGE_THRESHOLD:-0.6}"

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
  bash "$VLLM_SCRIPT"
  wait_for_vllm
}

use_local_llm() {
  export OPENAI_API_KEY="$OPENAI_API_KEY_LOCAL"
  export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
  export OPENAI_API_BASE="$OPENAI_BASE_URL"
  export OPENAI_DISABLE_THINKING="${OPENAI_DISABLE_THINKING:-1}"
  export PHYSICSVERIFIER_LLM_CONTEXT_TOKENS="${PHYSICSVERIFIER_LLM_CONTEXT_TOKENS:-32768}"
}

use_api_llm() {
  set -a
  # shellcheck disable=SC1091
  source "$ROOT/.env"
  set +a
  unset OPENAI_DISABLE_THINKING || true
}

run_location_eval() {
  local out_dir="$1"
  local label="$2"
  log "location-match evaluation ($label) iou=$LOCATION_IOU_THRESHOLD coverage=$LOCATION_COVERAGE_THRESHOLD"
  "$PYTHON" scripts/evaluate_physics_eval_sets.py \
    --dataset "$DATASET" \
    --results "$out_dir/checker_results.json" \
    --audit "$out_dir/checker_results.json" \
    --output "$out_dir/error_metrics.json" \
    --match-mode location \
    --location-iou-threshold "$LOCATION_IOU_THRESHOLD" \
    --location-coverage-threshold "$LOCATION_COVERAGE_THRESHOLD" \
    2>&1 | tee "$out_dir/location_eval.log"
}

run_local_baseline() {
  local out_dir="$OUT_ROOT/local_30b"
  mkdir -p "$out_dir"
  [[ -f "$DATASET" ]] || { log "ERROR: missing dataset $DATASET"; exit 2; }
  start_vllm_if_needed
  use_local_llm
  log "local pure LLM checker model=$LOCAL_MODEL dataset=$DATASET"
  "$PYTHON" scripts/run_llm_checker_baseline.py \
    --input "$DATASET" \
    --model "$LOCAL_MODEL" \
    --out_json "$out_dir/checker_results.json" \
    --flush-every 1 \
    --progress-interval 10 \
    --timeout 180 \
    2>&1 | tee "$out_dir/checker.log"
  run_location_eval "$out_dir" "local_30b"
}

run_upper_bound() {
  local out_dir="$OUT_ROOT/gemini_flash"
  mkdir -p "$out_dir"
  [[ -f "$DATASET" ]] || { log "ERROR: missing dataset $DATASET"; exit 2; }
  use_api_llm
  log "upper-bound pure LLM checker model=$UPPER_MODEL dataset=$DATASET"
  "$PYTHON" scripts/run_llm_checker_baseline.py \
    --input "$DATASET" \
    --model "$UPPER_MODEL" \
    --out_json "$out_dir/checker_results.json" \
    --flush-every 1 \
    --progress-interval 5 \
    --timeout 180 \
    2>&1 | tee "$out_dir/checker.log"
  run_location_eval "$out_dir" "gemini_flash"
}

run_eval_only() {
  local target="${EVAL_TARGET:-all}"
  if [[ "$target" == "all" || "$target" == "local" ]]; then
    run_location_eval "$OUT_ROOT/local_30b" "local_30b"
  fi
  if [[ "$target" == "all" || "$target" == "upper" ]]; then
    run_location_eval "$OUT_ROOT/gemini_flash" "gemini_flash"
  fi
}

summarize() {
  "$PYTHON" - <<PY
import json
from pathlib import Path
root = Path("$OUT_ROOT")
rows = []
for label, sub, checker in [
    ("local_30b_baseline", "local_30b", "$LOCAL_MODEL"),
    ("gemini_flash_upper_bound", "gemini_flash", "$UPPER_MODEL"),
]:
    p = root / sub / "error_metrics.json"
    if not p.exists():
        continue
    m = json.loads(p.read_text(encoding="utf-8")).get("summary", {})
    rows.append({
        "label": label,
        "checker_model": checker,
        "match_mode": m.get("match_mode"),
        "recall": m.get("recall"),
        "precision": m.get("precision"),
        "f1": m.get("f1"),
        "location_iou_threshold": $LOCATION_IOU_THRESHOLD,
        "location_coverage_threshold": $LOCATION_COVERAGE_THRESHOLD,
        "unmatched_pred_findings": m.get("location_unmatched_pred_findings"),
        "unmatched_gt_errors": m.get("location_unmatched_gt_errors"),
        "dataset_size": m.get("dataset_size"),
        "total_gt_errors": m.get("total_gt_errors"),
    })
out = root / "comparison_summary.json"
out.write_text(
    json.dumps(
        {
            "dataset": "$DATASET",
            "eval_match_mode": "location",
            "aligned_with": "scale_1500_cleaned",
            "variants": rows,
        },
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)
print(out.read_text(encoding="utf-8"))
PY
}

main() {
  mkdir -p "$OUT_ROOT/logs"
  case "$PHASE" in
    local) run_local_baseline ;;
    upper) run_upper_bound ;;
    eval) run_eval_only ;;
    all)
      [[ "$RUN_LOCAL" == "1" ]] && run_local_baseline
      [[ "$RUN_UPPER" == "1" ]] && run_upper_bound
      ;;
    *) log "ERROR: unknown PHASE=$PHASE (use local|upper|eval|all)"; exit 2 ;;
  esac
  summarize
  log "done -> $OUT_ROOT/comparison_summary.json"
}

main "$@"
