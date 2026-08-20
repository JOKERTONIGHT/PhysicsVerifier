#!/usr/bin/env bash
# Rebuild 1200-scale catalog incrementally from frozen 900 artifacts, then eval once.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-data/derived/expansion3000_scale_seed20260508}"
SEMANTIC_FULL="${SEMANTIC_FULL:-catalogs/semantic_experience.json}"
SOURCE_EXPANSION="${SOURCE_EXPANSION:-data/evaluation_sample_3000_expansion.json}"
ERROR_DATASET="${ERROR_DATASET:-$DATA_DIR/error_eval_dataset_100.json}"

RESULT_ROOT="${RESULT_ROOT:-results/scale_curve_error_v3_local_fp}"
CATALOG_ROOT="${CATALOG_ROOT:-catalogs/scale_curve_error_v3_local_fp}"
BASE_SCALE="${BASE_SCALE:-900}"
TARGET_SCALE="${TARGET_SCALE:-1200}"

BASE_TAG="scale_$(printf '%04d' "$BASE_SCALE")"
TARGET_TAG="scale_$(printf '%04d' "$TARGET_SCALE")"
STABLE_TAG="${TARGET_TAG}_stable"

BASE_CATALOG="$CATALOG_ROOT/rules_unified_${BASE_TAG}.json"
BASE_ARTIFACTS="$RESULT_ROOT/${BASE_TAG}/catalog_build"
OUT_CATALOG="$CATALOG_ROOT/rules_unified_${STABLE_TAG}.json"
OUT_DIR="$RESULT_ROOT/${STABLE_TAG}"
WORK_DIR="$OUT_DIR/catalog_build"

MODEL="${MODEL:-qwen3-30b-a3b-instruct-2507}"
VLLM_SCRIPT="${VLLM_SCRIPT:-/home/jinjianhan/deploy/qwen3_30b/start_vllm_30b.sh}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8766}"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-32768}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-local-vllm}"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

preflight() {
  for f in "$BASE_CATALOG" "$BASE_ARTIFACTS/cluster_proposals.json" "$SEMANTIC_FULL" "$ERROR_DATASET"; do
    if [[ ! -f "$f" ]]; then
      log "ERROR: missing required file: $f"
      exit 2
    fi
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

build_stable_catalog() {
  log "building stable ${TARGET_SCALE} catalog from ${BASE_SCALE} baseline"
  mkdir -p "$WORK_DIR" "$(dirname "$OUT_CATALOG")"
  "$PYTHON" scripts/build_scale_unified_catalog.py \
    --semantic-input "$SEMANTIC_FULL" \
    --expansion-input "$SOURCE_EXPANSION" \
    --expansion-size "$TARGET_SCALE" \
    --work-dir "$WORK_DIR" \
    --output "$OUT_CATALOG" \
    --baseline-catalog "$BASE_CATALOG" \
    --reuse-artifacts-dir "$BASE_ARTIFACTS"
  log "stable catalog -> $OUT_CATALOG"
}

run_eval() {
  log "running error-level eval for stable ${TARGET_SCALE}"
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
    --unified-catalog "$OUT_CATALOG" \
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
c=json.load(open("$OUT_CATALOG"))
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
s["build_mode"]="stable_incremental_from_${BASE_SCALE}"
json.dump(m, open("$OUT_DIR/error_metrics.json","w"), ensure_ascii=False, indent=2)
print(json.dumps(s, ensure_ascii=False, indent=2))
PY
  log "eval done -> $OUT_DIR/error_metrics.json wall_s=${wall}s"
}

preflight
build_stable_catalog
run_eval
log "all done stable catalog=$OUT_CATALOG eval=$OUT_DIR/error_metrics.json"
