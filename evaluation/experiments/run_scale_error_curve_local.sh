#!/usr/bin/env bash
# Leak-free rule-library scale curve: local vLLM or remote OpenAI-compatible API.
#
# Usage (local vLLM, detach-safe):
#   cd /home/jinjianhan/PhysicsVerifier
#   nohup bash scripts/run_scale_error_curve_local.sh \
#     > results/_scale_error_curve_local_batch.log 2>&1 &
#   echo $! > results/_scale_error_curve_local_batch.pid
#
# Usage (remote API — recommended when local vLLM context/trigger rate is unstable):
#   LLM_BACKEND=api RESULT_ROOT=results/scale_curve_error_v3_api \
#     nohup bash scripts/run_scale_error_curve_local.sh \
#     > results/_scale_error_curve_v3_api_batch.log 2>&1 &
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VLLM_MAX_LEN="${VLLM_MAX_LEN:-32768}"
PHYSICSVERIFIER_LLM_CONTEXT_TOKENS="${PHYSICSVERIFIER_LLM_CONTEXT_TOKENS:-32768}"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DATA_DIR="${DATA_DIR:-data/derived/expansion3000_scale_seed20260508}"
EXPANSION_POOL="${EXPANSION_POOL:-$DATA_DIR/expansion_pool.json}"
EVAL_HOLDOUT="${EVAL_HOLDOUT:-$DATA_DIR/eval_holdout_200.json}"
ERROR_DATASET="${ERROR_DATASET:-$DATA_DIR/error_eval_dataset_100.json}"
SPLIT_MANIFEST="${SPLIT_MANIFEST:-$DATA_DIR/split_manifest.json}"
SEMANTIC_FULL="${SEMANTIC_FULL:-catalogs/semantic_experience.json}"
DISTILLED_FULL="${DISTILLED_FULL:-results/scale_curve_error/semantic_experience_distilled_full.json}"
LLM_BACKEND="${LLM_BACKEND:-local}"  # local | api
RESULT_ROOT="${RESULT_ROOT:-results/scale_curve_error_v3}"
CATALOG_ROOT="${CATALOG_ROOT:-catalogs/scale_curve_error_v3}"
REPORT_OUTPUT="${REPORT_OUTPUT:-docs/规则库规模曲线实验报告_v3.md}"
STAMP_FILE="${STAMP_FILE:-$ROOT/results/_scale_error_curve_local_stamp.txt}"
SEED_BLUEPRINTS="${SEED_BLUEPRINTS:-catalogs/scenario_cluster_blueprints.json}"
FULL_CATALOG_BUILD="${FULL_CATALOG_BUILD:-1}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-large}"
CLUSTER_MAX_TOPICS="${CLUSTER_MAX_TOPICS:-0}"
CLUSTER_MIN_RULE_COUNT="${CLUSTER_MIN_RULE_COUNT:-4}"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
SOURCE_EXPANSION="${SOURCE_EXPANSION:-data/evaluation_sample_3000_expansion.json}"
ANNOTATED_ERROR_EVAL="${ANNOTATED_ERROR_EVAL:-data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain/error_eval_dataset_100.json}"
ERROR_EVAL_SIZE="${ERROR_EVAL_SIZE:-100}"

MODEL="${MODEL:-qwen3-30b-a3b-instruct-2507}"
CLUSTER_MODEL="${CLUSTER_MODEL:-$MODEL}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8766}"
VLLM_SCRIPT="${VLLM_SCRIPT:-/home/jinjianhan/deploy/qwen3_30b/start_vllm_30b.sh}"
MODEL_DIR="${MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
MODEL_MIN_BYTES="${MODEL_MIN_BYTES:-55000000000}"
MODEL_SHARD_COUNT="${MODEL_SHARD_COUNT:-16}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-local-vllm}"
LLM_API_KEY=""
LLM_BASE_URL=""
MANIFEST="${EXPERIENCE_CODE_MANIFEST:-results/experience_symbolic_program_manifest_v2_unified.json}"
MODULE="${EXPERIENCE_CODE_MODULE:-symbolic.generated_experience_checks_v2_unified}"

SCALES="${SCALES:-300,600,900,1200,1500,1800,2100,2400,2700}"
MIN_F1_GAIN="${MIN_F1_GAIN:-0.005}"
PATIENCE="${PATIENCE:-2}"
SKIP_SEMANTIC="${SKIP_SEMANTIC:-1}"
SKIP_EVAL="${SKIP_EVAL:-0}"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
echo "$STAMP" > "$STAMP_FILE"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

load_cloud_openai_from_dotenv() {
  # Preserve remote OpenAI credentials for embedding clustering before local vLLM overrides.
  if [[ -z "${EMBEDDING_OPENAI_API_KEY:-}" || -z "${EMBEDDING_OPENAI_BASE_URL:-}" ]]; then
    if [[ -f "$ENV_FILE" ]]; then
      eval "$(
        "$PYTHON" - "$ENV_FILE" <<'PY'
import os
import shlex
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE")
values = {}
if env_path.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path, override=False)
    except ImportError:
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in keys and key not in os.environ:
                os.environ[key] = value.strip().strip('"').strip("'")
for key in keys:
    val = os.environ.get(key)
    if val:
        values[key] = val
if values.get("OPENAI_API_KEY"):
    print(f"export EMBEDDING_OPENAI_API_KEY={shlex.quote(values['OPENAI_API_KEY'])}")
base = values.get("OPENAI_BASE_URL") or values.get("OPENAI_API_BASE")
if base:
    print(f"export EMBEDDING_OPENAI_BASE_URL={shlex.quote(base.rstrip('/'))}")
PY
      )"
    fi
  fi
  if [[ -z "${EMBEDDING_OPENAI_API_KEY:-}" ]]; then
    log "ERROR: EMBEDDING_OPENAI_API_KEY missing (set in $ENV_FILE or shell env)"
    exit 3
  fi
  if [[ -z "${EMBEDDING_OPENAI_BASE_URL:-}" ]]; then
    log "ERROR: EMBEDDING_OPENAI_BASE_URL missing (set in $ENV_FILE or shell env)"
    exit 3
  fi
  log "embedding API configured via ${EMBEDDING_OPENAI_BASE_URL}"
}

load_repo_llm_from_dotenv() {
  if [[ ! -f "$ENV_FILE" ]]; then
    log "ERROR: LLM_BACKEND=api but missing $ENV_FILE"
    exit 3
  fi
  eval "$(
    "$PYTHON" - "$ENV_FILE" <<'PY'
import os
import shlex
import sys
from pathlib import Path

env_path = Path(sys.argv[1])
keys = ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE")
try:
    from dotenv import load_dotenv
    load_dotenv(env_path, override=True)
except ImportError:
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key in keys:
                os.environ[key] = value.strip().strip('"').strip("'")
for key in keys:
    val = os.environ.get(key)
    if val:
        print(f"export {key}={shlex.quote(val)}")
PY
  )"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log "ERROR: OPENAI_API_KEY missing in $ENV_FILE"
    exit 3
  fi
  LLM_API_KEY="$OPENAI_API_KEY"
  LLM_BASE_URL="${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}"
  LLM_BASE_URL="${LLM_BASE_URL%/}"
  if [[ -z "$LLM_BASE_URL" ]]; then
    log "ERROR: OPENAI_BASE_URL / OPENAI_API_BASE missing in $ENV_FILE"
    exit 3
  fi
  export OPENAI_API_KEY OPENAI_BASE_URL OPENAI_API_BASE="$LLM_BASE_URL"
  log "LLM API configured via ${LLM_BASE_URL}"
}

setup_llm_backend() {
  case "$LLM_BACKEND" in
    api)
      load_repo_llm_from_dotenv
      ;;
    local)
      wait_for_model_weights
      if [[ ! -f "$VLLM_SCRIPT" ]]; then
        log "ERROR: VLLM_SCRIPT not found: $VLLM_SCRIPT"
        exit 3
      fi
      MODEL_DIR="$MODEL_DIR" MAX_LEN="$VLLM_MAX_LEN" bash "$VLLM_SCRIPT"
      wait_for_vllm
      use_local_llm
      LLM_API_KEY="$OPENAI_API_KEY"
      LLM_BASE_URL="$OPENAI_BASE_URL"
      ;;
    *)
      log "ERROR: LLM_BACKEND must be 'local' or 'api', got: $LLM_BACKEND"
      exit 2
      ;;
  esac
  export PHYSICSVERIFIER_LLM_CONTEXT_TOKENS
}

wait_for_model_weights() {
  # shellcheck source=/dev/null
  source "$(dirname "$VLLM_SCRIPT")/model_paths.sh"
  local dir="${MODEL_DIR:-$QWEN30B_MODEL_DIR}"
  local min_bytes="${MODEL_MIN_BYTES:-$QWEN30B_MIN_BYTES}"
  while true; do
    local sz
    sz=$(du -sb "$dir" 2>/dev/null | awk '{print $1}')
    if qwen30b_weights_ready "$dir" "$min_bytes"; then
      log "model weights ready dir=${dir} size=${sz}"
      return 0
    fi
    log "waiting for 30B FP download... dir=${dir} size=${sz:-0}"
    sleep 120
  done
}

wait_for_vllm() {
  local url="http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
  if curl -sf "$url" >/dev/null 2>&1; then
    log "vLLM already ready: $url"
    return 0
  fi
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

use_local_llm() {
  export OPENAI_API_KEY="${OPENAI_API_KEY_LOCAL}"
  export OPENAI_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
  export OPENAI_API_BASE="$OPENAI_BASE_URL"
}

prepare_splits() {
  log "prepare leak-free splits from $SOURCE_EXPANSION (annotated eval: $ANNOTATED_ERROR_EVAL, size=$ERROR_EVAL_SIZE)"
  "$PYTHON" scripts/prepare_expansion3000_scale_splits.py \
    --source "$SOURCE_EXPANSION" \
    --annotated-error-eval "$ANNOTATED_ERROR_EVAL" \
    --outdir "$DATA_DIR" \
    --error-eval-size "$ERROR_EVAL_SIZE"
}

preflight_error_eval() {
  if [[ ! -f "$ERROR_DATASET" ]]; then
    log "ERROR: missing error eval dataset (100-sample): $ERROR_DATASET"
    exit 3
  fi
  if [[ ! -f "$SPLIT_MANIFEST" ]]; then
    log "ERROR: missing split manifest: $SPLIT_MANIFEST"
    exit 3
  fi
  "$PYTHON" - <<PY
import json, sys
manifest = json.load(open("$SPLIT_MANIFEST"))
audit = manifest.get("overlap_audit", {})
if not audit.get("passes"):
    print("[error] split overlap audit failed:", audit, file=sys.stderr)
    sys.exit(3)
if audit.get("error_eval_100_subset_of_holdout_200") is False:
    print("[error] error_eval_100 must be subset of eval_holdout_200", file=sys.stderr)
    sys.exit(3)
print("[ok] error_eval_100 leak-free; eval_holdout_200 reserved (unused in rule mining)")
PY
}

run_semantic_full() {
  if [[ "$SKIP_SEMANTIC" == "1" ]]; then
    if [[ ! -f "$SEMANTIC_FULL" ]]; then
      log "ERROR: SKIP_SEMANTIC=1 but missing $SEMANTIC_FULL"
      exit 3
    fi
    log "skip semantic extraction (reuse $SEMANTIC_FULL)"
    return 0
  fi
  log "semantic experience on full expansion pool"
  mkdir -p "$(dirname "$SEMANTIC_FULL")"
  "$PYTHON" scripts/generate_experience_rules.py \
    --input "$EXPANSION_POOL" \
    --rules-catalog catalogs/rules_catalog_top_down.json \
    --model "$MODEL" \
    --output "$SEMANTIC_FULL" \
    --distilled-output "$DISTILLED_FULL" \
    --max-rules-per-sample 2 \
    --min-rule-count 1 \
    --resume
}

run_scale_point() {
  local n="$1"
  local tag="scale_$(printf '%04d' "$n")"
  local ckpt_input="$DATA_DIR/checkpoints/expansion_sample_$(printf '%04d' "$n").json"
  local out="$RESULT_ROOT/$tag"
  local sem="$out/semantic_experience.json"
  local dist="$out/semantic_experience_distilled.json"
  local catalog="$CATALOG_ROOT/rules_unified_${tag}.json"

  mkdir -p "$out" "$CATALOG_ROOT"

  "$PYTHON" scripts/subset_semantic_experience_for_scale.py \
    --semantic-input "$SEMANTIC_FULL" \
    --expansion-input "$ckpt_input" \
    --expansion-size "$n" \
    --semantic-output "$sem" \
    --distilled-output "$dist" \
    --min-rule-count 1

  if [[ "$FULL_CATALOG_BUILD" == "1" ]]; then
    "$PYTHON" scripts/build_scale_unified_catalog.py \
      --semantic-input "$SEMANTIC_FULL" \
      --expansion-input "$ckpt_input" \
      --expansion-size "$n" \
      --work-dir "$out/catalog_build" \
      --output "$catalog" \
      --seed-blueprints "$SEED_BLUEPRINTS" \
      --embedding-model "$EMBEDDING_MODEL" \
      --embedding-api-key "$EMBEDDING_OPENAI_API_KEY" \
      --embedding-base-url "$EMBEDDING_OPENAI_BASE_URL" \
      --cluster-model "$CLUSTER_MODEL" \
      --cluster-api-key "$LLM_API_KEY" \
      --cluster-base-url "$LLM_BASE_URL" \
      --cluster-max-topics "$CLUSTER_MAX_TOPICS" \
      --cluster-min-rule-count "$CLUSTER_MIN_RULE_COUNT"
  else
    "$PYTHON" scripts/build_unified_catalog.py \
      --experience-distilled "$dist" \
      --scenario-cluster-blueprints "$SEED_BLUEPRINTS" \
      --output "$catalog"
  fi

  if [[ "$SKIP_EVAL" == "1" ]]; then
    return 0
  fi

  cp -f "$ERROR_DATASET" "$out/error_eval_dataset_100.json"
  local t0 t1 wall
  t0=$(date -u +%s)
  "$PYTHON" scripts/run_verifier.py \
    --input "$out/error_eval_dataset_100.json" \
    --output "$out/error_verifier_results.json" \
    --symbolic-output "$out/error_symbolic_audit.json" \
    --model "$MODEL" \
    --unified-catalog "$catalog" \
    --experience-code-manifest "$MANIFEST" \
    --experience-code-module "$MODULE" \
    --max-per-sample 12 \
    --max-per-paragraph 2 \
    --progress-interval 10 \
    --no-symbolic-check \
    2>&1 | tee "$out/run_verifier.log"

  "$PYTHON" scripts/evaluate_physics_eval_sets.py \
    --dataset "$out/error_eval_dataset_100.json" \
    --results "$out/error_verifier_results.json" \
    --audit "$out/error_symbolic_audit.json" \
    --output "$out/error_metrics.json" \
    --match-mode location

  t1=$(date -u +%s)
  wall=$((t1 - t0))
  local rules
  rules=$("$PYTHON" - <<PY
import json
c=json.load(open("$catalog"))
print(c.get("metadata",{}).get("total_executable_rules",0))
PY
)
  "$PYTHON" - <<PY
import json
p="$out/error_metrics.json"
m=json.load(open(p))
s=m.get("summary",m)
s["expansion_size"]=$n
s["catalog_rules"]=$rules
s["wall_s"]=$wall
json.dump(m, open(p,"w"), ensure_ascii=False, indent=2)
PY
  log "scale=$n rules=$rules metrics=$out/error_metrics.json wall_s=${wall}s"
}

read_f1() {
  local metrics="$1"
  "$PYTHON" - <<PY
import json
m=json.load(open("$metrics"))
s=m.get("summary",m)
print(float(s.get("f1") or 0.0))
PY
}

aggregate_and_plot() {
  local plot_title
  if [[ "$LLM_BACKEND" == "api" ]]; then
    plot_title="Error-Level Metrics vs Expansion Size (Per-Scale Full Pipeline, Remote API 30B)"
  else
    plot_title="Error-Level Metrics vs Expansion Size (Per-Scale Full Pipeline, Local 30B)"
  fi
  "$PYTHON" scripts/aggregate_error_scale_curve.py \
    --metrics-glob "$RESULT_ROOT/scale_*/error_metrics.json" \
    --output-csv "$RESULT_ROOT/curve_metrics.csv" \
    --output-json "$RESULT_ROOT/curve_metrics.json"
  "$PYTHON" scripts/plot_error_scale_curve.py \
    --input-csv "$RESULT_ROOT/curve_metrics.csv" \
    --output "$RESULT_ROOT/error_scale_curve.png" \
    --title "$plot_title"
}

# ---- main ----
log "scale error curve batch STAMP=$STAMP LLM_BACKEND=$LLM_BACKEND RESULT_ROOT=$RESULT_ROOT"
prepare_splits
load_cloud_openai_from_dotenv
setup_llm_backend

preflight_error_eval
run_semantic_full

prev_f1=""
bad_streak=0
IFS=',' read -r -a scale_arr <<< "$SCALES"
for n in "${scale_arr[@]}"; do
  n="${n// /}"
  [[ -z "$n" ]] && continue
  if [[ ! -f "$DATA_DIR/checkpoints/expansion_sample_$(printf '%04d' "$n").json" ]]; then
    log "skip scale=$n (checkpoint file missing)"
    continue
  fi
  run_scale_point "$n"
  if [[ "$SKIP_EVAL" == "1" ]]; then
    continue
  fi
  cur_f1="$(read_f1 "$RESULT_ROOT/scale_$(printf '%04d' "$n")/error_metrics.json")"
  if [[ -n "$prev_f1" ]]; then
    gain=$("$PYTHON" - <<PY
prev=float("$prev_f1"); cur=float("$cur_f1"); print(cur-prev)
PY
)
    log "scale=$n f1=$cur_f1 gain_vs_prev=$gain"
    awk -v g="$gain" -v m="$MIN_F1_GAIN" 'BEGIN{ exit !(g < m) }' && bad_streak=$((bad_streak + 1)) || bad_streak=0
    if [[ "$bad_streak" -ge "$PATIENCE" ]]; then
      log "adaptive stop: F1 gain < $MIN_F1_GAIN for $PATIENCE consecutive scales"
      break
    fi
  else
    log "scale=$n f1=$cur_f1 (baseline)"
  fi
  prev_f1="$cur_f1"
done

aggregate_and_plot
"$PYTHON" scripts/summarize_scale_error_curve_report.py \
  --result-root "$RESULT_ROOT" \
  --split-manifest "$SPLIT_MANIFEST" \
  --stamp-file "$STAMP_FILE" \
  --llm-backend "$LLM_BACKEND" \
  --output "$REPORT_OUTPUT"
log "done STAMP=$STAMP results=$RESULT_ROOT report=$REPORT_OUTPUT"
