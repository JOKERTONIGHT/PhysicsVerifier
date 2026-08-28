#!/usr/bin/env bash
# Start PhysicsVerifier reward server with local judge or external OpenAI-compatible API.
set -euo pipefail

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
VENV="${VENV:-${ROOT}/.venv}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8770}"
LOG="${LOG:-${ROOT}/logs/physics_reward_server.log}"
PID_FILE="${PID_FILE:-${ROOT}/logs/physics_reward_server.pid}"

mkdir -p "$(dirname "$LOG")"

cd "${ROOT}" || exit 1
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${ROOT}/.env"
  set +a
fi
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PHYSICS_ROOT="${ROOT}"
export PHYSICS_REWARD_LAMBDA="${PHYSICS_REWARD_LAMBDA:-0.3}"
export PHYSICS_REWARD_ERROR_CAP="${PHYSICS_REWARD_ERROR_CAP:-3}"
export PHYSICS_REWARD_CONCURRENCY="${PHYSICS_REWARD_CONCURRENCY:-8}"
export PHYSICS_REWARD_MAX_RESPONSE_CHARS="${PHYSICS_REWARD_MAX_RESPONSE_CHARS:-12000}"
export PHYSICS_REWARD_MODE="${PHYSICS_REWARD_MODE:-answer_low_verifier}"
export PHYSICS_REWARD_W_ANSWER="${PHYSICS_REWARD_W_ANSWER:-1.0}"
export PHYSICS_REWARD_W_FORMAT="${PHYSICS_REWARD_W_FORMAT:-0.05}"
export PHYSICS_REWARD_W_VERIFIER="${PHYSICS_REWARD_W_VERIFIER:-0.1}"
export PHYSICS_VERIFIER_SAMPLE_RATE="${PHYSICS_VERIFIER_SAMPLE_RATE:-1.0}"
if [[ "${PHYSICS_REWARD_MODE}" == "process_paragraph" ]]; then
  export PHYSICS_REWARD_VERIFIER_ON_WRONG="${PHYSICS_REWARD_VERIFIER_ON_WRONG:-1}"
  export PHYSICS_REWARD_W_ANSWER=0
  export PHYSICS_REWARD_W_FORMAT=0
  export PHYSICS_REWARD_W_CLEAN="${PHYSICS_REWARD_W_CLEAN:-0.5}"
  export PHYSICS_REWARD_W_FIRST="${PHYSICS_REWARD_W_FIRST:-0.3}"
  export PHYSICS_REWARD_W_DENSE="${PHYSICS_REWARD_W_DENSE:-0.2}"
  export PHYSICS_REWARD_CONCURRENCY="${PHYSICS_REWARD_CONCURRENCY:-12}"
  export PHYSICS_REWARD_MAX_RESPONSE_CHARS="${PHYSICS_REWARD_MAX_RESPONSE_CHARS:-3072}"
  export PHYSICSVERIFIER_UNIFIED_RULE_TOP_N="${PHYSICSVERIFIER_UNIFIED_RULE_TOP_N:-4}"
  export PHYSICSVERIFIER_PRECISION_MODE="${PHYSICSVERIFIER_PRECISION_MODE:-balanced}"
  export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE="${PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE:-lexical}"
fi
if [[ "${PHYSICS_REWARD_MODE}" == "llm_step_score" ]]; then
  export PHYSICSVERIFIER_LLM_MODEL="${PHYSICSVERIFIER_LLM_MODEL:-deepseek-v4-flash}"
  export LLM_STEP_JUDGE_TIMEOUT="${LLM_STEP_JUDGE_TIMEOUT:-300}"
  export LLM_STEP_JUDGE_CONCURRENCY="${LLM_STEP_JUDGE_CONCURRENCY:-32}"
  export PHYSICS_REWARD_CONCURRENCY="${PHYSICS_REWARD_CONCURRENCY:-32}"
  export PHYSICS_REWARD_W_ANSWER=0
  export PHYSICS_REWARD_W_FORMAT=0
  export PHYSICS_REWARD_W_VERIFIER=0
  if [[ -z "${OPENAI_BASE_URL:-}" || "${OPENAI_BASE_URL}" == *"127.0.0.1"* ]]; then
    echo "[error] llm_step_score requires remote OPENAI_BASE_URL from .env (not a local judge)" >&2
    exit 2
  fi
  if [[ -z "${OPENAI_API_KEY:-}" || "${OPENAI_API_KEY}" == "EMPTY" ]]; then
    echo "[error] llm_step_score requires OPENAI_API_KEY from .env" >&2
    exit 2
  fi
fi

CONFIG_SIG="${PHYSICS_REWARD_MODE}|c${PHYSICS_REWARD_CONCURRENCY}|m${PHYSICSVERIFIER_LLM_MODEL:-}|pvllm_step_v1|t${LLM_STEP_JUDGE_TIMEOUT:-180}|mt${LLM_STEP_JUDGE_MAX_TOKENS:-4096}|retries${LLM_STEP_JUDGE_MAX_RETRIES:-6}|jc${LLM_STEP_JUDGE_CONCURRENCY:-32}|par1|jr2|top${PHYSICSVERIFIER_UNIFIED_RULE_TOP_N:-4}|max${PHYSICS_REWARD_MAX_RESPONSE_CHARS}|prec${PHYSICSVERIFIER_PRECISION_MODE:-strict}|ret${PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE:-semantic}|cache${PHYSICS_REWARD_CACHE_SIZE:-4096}|url${OPENAI_BASE_URL:-}"
CONFIG_FILE="$(dirname "$PID_FILE")/physics_reward_server.config"

CONFIGURED_OPENAI_BASE_URL="${PHYSICSVERIFIER_OPENAI_BASE_URL:-${OPENAI_BASE_URL:-}}"
if [[ "${PHYSICS_REWARD_MODE}" != "llm_step_score" && -n "${PHYSICSVERIFIER_OPENAI_BASE_URL:-}" ]]; then
  export OPENAI_BASE_URL="${PHYSICSVERIFIER_OPENAI_BASE_URL}"
fi
if [[ "${PHYSICS_REWARD_MODE}" != "llm_step_score" && -n "${PHYSICSVERIFIER_OPENAI_API_KEY:-}" ]]; then
  export OPENAI_API_KEY="${PHYSICSVERIFIER_OPENAI_API_KEY}"
fi
if [[ "${PHYSICS_REWARD_MODE}" == "llm_step_score" ]]; then
  if [[ -z "${OPENAI_BASE_URL:-}" || "${OPENAI_BASE_URL}" == *"127.0.0.1"* ]]; then
    echo "[error] llm_step_score requires remote OPENAI_BASE_URL from .env (not a local judge)" >&2
    exit 2
  fi
  export PHYSICSVERIFIER_LLM_MODEL="${PHYSICSVERIFIER_LLM_MODEL:-deepseek-v4-flash}"
else
  export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
  export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8766/v1}"
  export PHYSICSVERIFIER_LLM_MODEL="${PHYSICSVERIFIER_LLM_MODEL:-qwen3-30b-a3b}"
fi
export PHYSICSVERIFIER_UNIFIED_RULES="${PHYSICSVERIFIER_UNIFIED_RULES:-${ROOT}/catalogs/rules_unified_3000_runtime_backfilled.json}"
export PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE="${PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE:-semantic}"
# The norm_* runtime catalog has no matching exp_* symbolic manifest.
export PHYSICSVERIFIER_SYMBOLIC_ENABLED="${PHYSICSVERIFIER_SYMBOLIC_ENABLED:-0}"

port_pid() {
  ss -lptn "sport = :${PORT}" 2>/dev/null | sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -n1
}

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "$old_pid" ]] && kill -0 "$old_pid" 2>/dev/null; then
    if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      old_sig="$(cat "${CONFIG_FILE}" 2>/dev/null || true)"
      if [[ "${old_sig}" == "${CONFIG_SIG}" ]]; then
        echo "[ok] reward server already running pid=$old_pid config=${CONFIG_SIG}"
        exit 0
      fi
      echo "[reward] config changed (${old_sig:-none} -> ${CONFIG_SIG}); restarting"
      kill -TERM "$old_pid" 2>/dev/null || true
      sleep 2
      kill -9 "$old_pid" 2>/dev/null || true
    fi
  fi
fi

if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
  occupant="$(port_pid)"
  old_sig="$(cat "${CONFIG_FILE}" 2>/dev/null || true)"
  if [[ "${old_sig}" == "${CONFIG_SIG}" && -n "${occupant}" ]] && kill -0 "${occupant}" 2>/dev/null; then
    echo "${occupant}" >"$PID_FILE"
    echo "[ok] reward server already running pid=${occupant} (recovered from ${HOST}:${PORT})"
    exit 0
  fi
  echo "[reward] ${HOST}:${PORT} occupied by pid=${occupant:-unknown}; restarting for config=${CONFIG_SIG}"
  if [[ -n "${occupant}" ]]; then
    kill -TERM "${occupant}" 2>/dev/null || true
    sleep 2
    kill -9 "${occupant}" 2>/dev/null || true
  fi
fi

if [[ "${PHYSICS_REWARD_MODE}" == "llm_step_score" && "${SKIP_LLM_PREFLIGHT:-0}" != "1" ]]; then
  echo "[reward] llm_step_score remote API at ${OPENAI_BASE_URL} model=${PHYSICSVERIFIER_LLM_MODEL}"
  "${VENV}/bin/python" - <<'PY'
import os, sys
sys.path.insert(0, os.environ.get("PHYSICS_ROOT", "/home/jinjianhan/PhysicsVerifier"))
from training.reward_server.llm_step_judge import DEFAULT_MODEL, LLMStepJudge, require_remote_model
model = os.environ.get("PHYSICSVERIFIER_LLM_MODEL", DEFAULT_MODEL)
if model != DEFAULT_MODEL:
    print(f"[error] refusing model fallback: {model} (required {DEFAULT_MODEL})", file=sys.stderr)
    sys.exit(2)
require_remote_model(model)
judge = LLMStepJudge.from_env()
judge.score_group(
    "A mass m is at rest on a frictionless table. What is its acceleration?",
    ["Net force is zero so a=0.", "The answer is 42 without derivation."],
)
print("[ok] llm_step_score preflight passed")
PY
elif [[ "${PHYSICS_REWARD_MODE}" == "llm_step_score" ]]; then
  echo "[reward] llm_step_score skipping preflight; remote API at ${OPENAI_BASE_URL} model=${PHYSICSVERIFIER_LLM_MODEL}"
elif [[ -n "${CONFIGURED_OPENAI_BASE_URL}" ]]; then
  echo "[reward] using external verifier API at ${OPENAI_BASE_URL}"
  "${VENV}/bin/python" - <<'PY'
import os, sys
from openai import OpenAI
base = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
key = os.environ.get("OPENAI_API_KEY", "EMPTY")
model = os.environ.get("PHYSICSVERIFIER_LLM_MODEL", "")
client = OpenAI(base_url=base, api_key=key)
models = [m.id for m in client.models.list().data]
if model and model not in models:
    print(f"[warn] configured model {model} not in remote list; available={models[:5]}", file=sys.stderr)
print("[ok] external verifier API reachable")
PY
elif [[ "${PHYSICS_REWARD_MODE}" == "answer_only" ]]; then
  echo "[reward] answer_only mode; skipping local judge/API warmup"
else
  curl -sf "${OPENAI_BASE_URL%/}/models" >/dev/null || {
    echo "[error] local judge unavailable at ${OPENAI_BASE_URL}; set PHYSICSVERIFIER_OPENAI_BASE_URL or PHYSICS_REWARD_MODE=answer_only" >&2
    exit 2
  }
fi

nohup "${VENV}/bin/python" "${ROOT}/training/reward_server/physics_reward_server.py" \
  --host "$HOST" --port "$PORT" \
  --reward-mode "${PHYSICS_REWARD_MODE}" \
  --concurrency "${PHYSICS_REWARD_CONCURRENCY}" \
  >"$LOG" 2>&1 &
echo $! >"$PID_FILE"
printf '%s' "${CONFIG_SIG}" >"${CONFIG_FILE}"
ready=0
for _ in $(seq 1 20); do
  if curl -sf "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" -ne 1 ]]; then
  echo "[error] reward server failed to become healthy on ${HOST}:${PORT}; see ${LOG}" >&2
  tail -n 40 "${LOG}" >&2 || true
  exit 2
fi
echo "[ok] reward server started on ${HOST}:${PORT} mode=${PHYSICS_REWARD_MODE}"
