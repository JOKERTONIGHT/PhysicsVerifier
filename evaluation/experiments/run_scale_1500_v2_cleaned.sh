#!/usr/bin/env bash
# Re-run local 30B error-level eval at scale 1500 on cleaned v2 dataset.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-data/derived/expansion3000_scale_v2_eval_deepseek_seed20260508}"
ERROR_DATASET="${ERROR_DATASET:-data/derived/eval_v2_deepseek_v4_pro_seed20260508/error_eval_dataset_100.cleaned.json}"
RESULT_ROOT="${RESULT_ROOT:-results/scale_curve_error_v2_local_30b}"
CATALOG_ROOT="${CATALOG_ROOT:-catalogs/scale_curve_error_v2_local_30b}"
TARGET_SCALE="${TARGET_SCALE:-1500}"
TARGET_TAG="scale_$(printf '%04d' "$TARGET_SCALE")"
OUT_TAG="${OUT_TAG:-${TARGET_TAG}_cleaned}"
OUT_DIR="$RESULT_ROOT/$OUT_TAG"
CATALOG="$CATALOG_ROOT/rules_unified_${TARGET_TAG}.json"

MODEL="${MODEL:-qwen3-30b-a3b-instruct-2507}"
VLLM_SCRIPT="${VLLM_SCRIPT:-/home/jinjianhan/deploy/qwen3_30b/start_vllm_30b.sh}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8766}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-32768}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-local-vllm}"
MANIFEST="${EXPERIENCE_CODE_MANIFEST:-results/experience_symbolic_program_manifest_v2_unified.json}"
MODULE="${EXPERIENCE_CODE_MODULE:-symbolic.generated_experience_checks_v2_unified}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

preflight() {
  for f in "$ERROR_DATASET" "$CATALOG"; do
    [[ -f "$f" ]] || { log "ERROR: missing $f"; exit 2; }
  done
}

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

run_eval() {
  log "scale=${TARGET_SCALE} cleaned eval dataset=$ERROR_DATASET"
  mkdir -p "$OUT_DIR"
  cp -f "$ERROR_DATASET" "$OUT_DIR/error_eval_dataset_100.json"
  start_vllm_if_needed
  use_local_llm
  local t0 t1 wall rules
  t0=$(date -u +%s)
  "$PYTHON" scripts/run_verifier.py \
    --input "$OUT_DIR/error_eval_dataset_100.json" \
    --output "$OUT_DIR/error_verifier_results.json" \
    --symbolic-output "$OUT_DIR/error_symbolic_audit.json" \
    --model "$MODEL" \
    --unified-catalog "$CATALOG" \
    --experience-code-manifest "$MANIFEST" \
    --experience-code-module "$MODULE" \
    --max-per-sample 12 \
    --max-per-paragraph 2 \
    --progress-interval 10 \
    --no-symbolic-check \
    2>&1 | tee "$OUT_DIR/run_verifier.log"
  "$PYTHON" scripts/evaluate_physics_eval_sets.py \
    --dataset "$OUT_DIR/error_eval_dataset_100.json" \
    --results "$OUT_DIR/error_verifier_results.json" \
    --audit "$OUT_DIR/error_symbolic_audit.json" \
    --output "$OUT_DIR/error_metrics.json" \
    --match-mode location
  t1=$(date -u +%s)
  wall=$((t1 - t0))
  rules=$("$PYTHON" - <<PY
import json
c=json.load(open("$CATALOG"))
print(c.get("metadata",{}).get("total_executable_rules",0))
PY
)
  "$PYTHON" - <<PY
import json
m=json.load(open("$OUT_DIR/error_metrics.json"))
s=m.get("summary",m)
s["expansion_size"]=$TARGET_SCALE
s["catalog_rules"]=$rules
s["wall_s"]=$wall
s["dataset"]="$ERROR_DATASET"
s["build_mode"]="cleaned_dataset_reuse_catalog"
json.dump(m, open("$OUT_DIR/error_metrics.json","w"), ensure_ascii=False, indent=2)
print(json.dumps(s, ensure_ascii=False, indent=2))
PY
  log "done -> $OUT_DIR/error_metrics.json wall_s=${wall}s"
}

preflight
run_eval
