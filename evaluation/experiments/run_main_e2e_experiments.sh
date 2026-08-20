#!/usr/bin/env bash
# Main experiment batch: 900-scale catalog, error-level eval WITH symbolic checks.
# Runs sequentially: 30B (API) → 4B (local vLLM) → 235B (API).
# All three use the same pipeline: unified catalog + experience-code symbolic + run_verifier.
#
# Usage (detach-safe):
#   cd /home/jinjianhan/PhysicsVerifier
#   nohup bash scripts/run_main_e2e_experiments.sh \
#     > results/_main_e2e_experiments_batch.log 2>&1 &
#   echo $! > results/_main_e2e_experiments_batch.pid
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source "$ROOT/evaluation/experiments/catalog_defaults.sh"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
VLLM_4B_SCRIPT="${VLLM_4B_SCRIPT:-/home/jinjianhan/deploy/qwen3_q4/start_vllm_4b.sh}"
VLLM_4B_HOST="${VLLM_4B_HOST:-127.0.0.1}"
VLLM_4B_PORT="${VLLM_4B_PORT:-8765}"
OPENAI_API_KEY_LOCAL="${OPENAI_API_KEY_LOCAL:-local-vllm}"

export PHYSICSVERIFIER_LLM_CONTEXT_TOKENS="${PHYSICSVERIFIER_LLM_CONTEXT_TOKENS:-32768}"

STAMP="${STAMP:-$(date -u +%Y%m%d_%H%M%S)}"
echo "$STAMP" > "$ROOT/results/_main_e2e_experiments_stamp.txt"

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

load_api_dotenv() {
  unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_API_BASE 2>/dev/null || true
  if [[ ! -f "$ENV_FILE" ]]; then
    log "ERROR: missing $ENV_FILE for API experiments"
    exit 3
  fi
  eval "$(
    "$PYTHON" - "$ENV_FILE" <<'PY'
import os, shlex, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(sys.argv[1]), override=True)
for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE"):
    val = os.environ.get(key)
    if val:
        print(f"export {key}={shlex.quote(val)}")
PY
  )"
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log "ERROR: OPENAI_API_KEY not set"
    exit 3
  fi
  log "API backend: ${OPENAI_BASE_URL:-${OPENAI_API_BASE:-default}}"
}

use_local_4b_llm() {
  export OPENAI_API_KEY="$OPENAI_API_KEY_LOCAL"
  export OPENAI_BASE_URL="http://${VLLM_4B_HOST}:${VLLM_4B_PORT}/v1"
  export OPENAI_API_BASE="$OPENAI_BASE_URL"
}

wait_for_vllm_4b() {
  local url="http://${VLLM_4B_HOST}:${VLLM_4B_PORT}/v1/models"
  local i=0
  while ! curl -sf "$url" >/dev/null 2>&1; do
    i=$((i + 1))
    if [[ "$i" -gt 90 ]]; then
      log "ERROR: 4B vLLM not ready at $url"
      exit 4
    fi
    sleep 5
  done
  log "4B vLLM ready: $url"
}

ensure_symbolic_module() {
  local min_checks="${SYMBOLIC_MIN_CHECKS:-1150}"
  local count=0 minimal=0
  if [[ -f "$MAIN_EXPERIENCE_MANIFEST" ]]; then
    read -r count minimal <<< "$("$PYTHON" - <<PY
import json
from pathlib import Path
import re
m=json.load(open("$MAIN_EXPERIENCE_MANIFEST"))
checks=m.get("checks") or []
mod=Path("symbolic/generated_experience_checks_0900.py")
src=mod.read_text(encoding="utf-8") if mod.exists() else ""
minimal=0
for c in checks:
    fn=c.get("function_name","")
    if not fn:
        continue
    mfn=re.search(rf"def {re.escape(fn)}\\(sample: dict\\).*?(?=\\ndef check_|\\nEXPERIENCE_CHECK_REGISTRY|$)", src, re.S)
    code=mfn.group(0) if mfn else ""
    if "触发经验规则(回退)" in code or ("keys = [" in code and "for key in keys:" in code):
        minimal+=1
print(len(checks), minimal)
PY
)"
  fi
  if [[ "$count" -ge "$min_checks" && "${minimal:-0}" -eq 0 ]]; then
    log "symbolic module ready: $count checks (v2-style)"
    return 0
  fi
  if [[ "${minimal:-0}" -gt 0 ]]; then
    log "upgrading $minimal minimal checks to v2-style structured template"
    bash "$ROOT/evaluation/experiments/generate_symbolic_checks_0900.sh" --fallback-only --refresh-fallback
  fi
  count=$("$PYTHON" - <<PY
import json
print(len(json.load(open("$MAIN_EXPERIENCE_MANIFEST")).get("checks") or []))
PY
)
  log "symbolic module ready: $count checks"
  if [[ "$count" -lt "$min_checks" ]]; then
    log "ERROR: expected >= $min_checks symbolic checks, got $count"
    exit 5
  fi
}

preflight_datasets() {
  for f in "$MAIN_ERROR_DATASET" "$MAIN_UNIFIED_CATALOG"; do
    if [[ ! -f "$f" ]]; then
      log "ERROR: missing required file: $f"
      exit 3
    fi
  done
}

run_full_e2e() {
  local tag="$1"
  local model="$2"
  local backend="$3"
  local out="$ROOT/results/${tag}_${STAMP}"
  mkdir -p "$out"
  log "starting e2e tag=$tag model=$model backend=$backend out=$out"
  log "  pipeline: error-level eval + symbolic=ON manifest=$MAIN_EXPERIENCE_MANIFEST module=$MAIN_EXPERIENCE_MODULE"

  RUN_TAG="$tag" \
  OUTDIR="$out" \
  CHECK_MODEL="$model" \
  STRONG_MODEL="$model" \
  UNIFIED_CATALOG="$MAIN_UNIFIED_CATALOG" \
  EXPERIENCE_CODE_MANIFEST="$MAIN_EXPERIENCE_MANIFEST" \
  EXPERIENCE_CODE_MODULE="$MAIN_EXPERIENCE_MODULE" \
  ERROR_DATASET="$MAIN_ERROR_DATASET" \
  SKIP_BUILD=1 \
  SKIP_QUESTION_EVAL=1 \
  NO_SYMBOLIC_CHECK=0 \
  bash "$ROOT/scripts/run_e2e_with_experience_symbolic.sh" \
    2>&1 | tee "$out/batch_wrapper.log"

  log "finished e2e tag=$tag metrics error=$out/error_metrics.json"
}

# ---- main ----
log "main e2e batch STAMP=$STAMP catalog=$MAIN_UNIFIED_CATALOG"
preflight_datasets
ensure_symbolic_module

# 1) 30B API error-level e2e (with symbolic)
load_api_dotenv
run_full_e2e "e2e_0900_30b_api_sym" "$MAIN_MODEL_30B" "api"

# 2) 4B local vLLM error-level e2e (with symbolic)
if [[ -x "$VLLM_4B_SCRIPT" || -f "$VLLM_4B_SCRIPT" ]]; then
  MAX_LEN="${VLLM_4B_MAX_LEN:-16384}" bash "$VLLM_4B_SCRIPT"
  wait_for_vllm_4b
  use_local_4b_llm
  run_full_e2e "e2e_0900_4b_local_sym" "$MAIN_MODEL_4B" "local_vllm"
else
  log "WARN: 4B vLLM script missing, skipping 4B experiment"
fi

# 3) 235B API error-level e2e (with symbolic)
load_api_dotenv
run_full_e2e "e2e_0900_235b_api_sym" "$MAIN_MODEL_235B" "api"

log "all main e2e experiments complete STAMP=$STAMP"
