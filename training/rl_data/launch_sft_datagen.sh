#!/usr/bin/env bash
# Start 3x Qwen3-30B-A3B on idle GPUs and generate SFT solutions via rejection sampling.
# Local first; API fill uses the first OPENAI_* pair in .env (line-2 key).
# Uses ports 8780–8783 so HiPhO eval on :8766 is undisturbed.
set -euo pipefail
ulimit -f unlimited 2>/dev/null || true
export TMPDIR="${TMPDIR:-/slow_share/jinjianhan/tmp/swift}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}"

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
MODEL_DIR="${JUDGE_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-30B-A3B-Instruct-2507}"
SERVED_NAME="${SFT_GEN_MODEL:-qwen3-30b-a3b}"
LB_PORT="${SFT_GEN_LB_PORT:-8780}"
PORTS=(8781 8782 8783)
RUN_IDS=(sft_gen0 sft_gen1 sft_gen2)
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
CKPT_LOG="${CKPT_LOG:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft/sft_datagen.log}"
mkdir -p "${LOG_DIR}" "$(dirname "${CKPT_LOG}")"
# shellcheck disable=SC1091
source "${ROOT}/training/swift/gpu_idle.sh"

start_one() {
  local gpu="$1" port="$2" run_id="$3"
  if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
    echo "[sft-gen] already ready gpu=${gpu} port=${port}"
    return 0
  fi
  RUN_ID="${run_id}" MODEL_DIR="${MODEL_DIR}" PORT="${port}" CUDA_DEVICE="${gpu}" \
    MAX_LEN=8192 GPU_UTIL="${GPU_UTIL:-0.88}" SERVED_NAME="${SERVED_NAME}" \
    LOG="${LOG_DIR}/${run_id}_vllm.log" PID_FILE="${LOG_DIR}/${run_id}_vllm.pid" \
    VLLM_READY_SECS=7200 ENABLE_PREFIX_CACHING=1 \
    bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" start
}

if [[ "${SKIP_LOCAL:-0}" != "1" ]]; then
  if curl -sf "http://127.0.0.1:${LB_PORT}/v1/models" >/dev/null 2>&1 \
      && curl -sf "http://127.0.0.1:${PORTS[0]}/v1/models" >/dev/null 2>&1; then
    echo "[sft-gen] reusing local generators on :${LB_PORT} / ${PORTS[*]}"
    GPUS=(reuse reuse reuse)
  else
    if [[ -n "${SFT_GEN_GPUS:-}" ]]; then
      IFS=',' read -r -a GPUS <<<"${SFT_GEN_GPUS}"
    else
      gpu_csv="$(wait_idle_csv 3)"
      IFS=',' read -r -a GPUS <<<"${gpu_csv}"
    fi
    if [[ "${#GPUS[@]}" -lt 3 ]]; then
      echo "[error] need 3 GPUs for SFT datagen, got ${SFT_GEN_GPUS:-${GPUS[*]}}" >&2
      exit 2
    fi
    echo "[sft-gen] gpus=${GPUS[0]},${GPUS[1]},${GPUS[2]}"
  fi

  start_pids=()
  for i in 0 1 2; do
    start_one "${GPUS[$i]}" "${PORTS[$i]}" "${RUN_IDS[$i]}" &
    start_pids+=($!)
  done
  fail=0
  for p in "${start_pids[@]}"; do
    wait "${p}" || fail=1
  done
  [[ "${fail}" -eq 0 ]] || { echo "[error] one or more 30B generators failed to start" >&2; exit 2; }

  if [[ -f "${LOG_DIR}/sft_gen_lb.pid" ]]; then
    old="$(cat "${LOG_DIR}/sft_gen_lb.pid" 2>/dev/null || true)"
    if [[ -n "${old}" ]] && kill -0 "${old}" 2>/dev/null; then
      kill -TERM "${old}" 2>/dev/null || true
      sleep 1
      kill -9 "${old}" 2>/dev/null || true
    fi
  fi
  nohup "${PYTHON}" "${ROOT}/training/openrlhf/judge_lb_proxy.py" \
    --host 127.0.0.1 --port "${LB_PORT}" \
    --backends "127.0.0.1:${PORTS[0]},127.0.0.1:${PORTS[1]},127.0.0.1:${PORTS[2]}" \
    >>"${LOG_DIR}/sft_gen_lb.log" 2>&1 &
  echo $! >"${LOG_DIR}/sft_gen_lb.pid"
  for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:${LB_PORT}/health" >/dev/null 2>&1 && break
    sleep 0.5
  done
  curl -sf "http://127.0.0.1:${LB_PORT}/v1/models" >/dev/null \
    || { echo "[error] SFT gen LB not ready on :${LB_PORT}" >&2; exit 2; }

  echo "[sft-gen] generating solutions via ${LB_PORT}"
  hint_args=()
  if [[ "${SFT_HINT_GOLD:-1}" == "1" ]]; then
    hint_args+=(--hint-gold)
  else
    hint_args+=(--no-hint-gold)
  fi
  "${PYTHON}" "${ROOT}/training/rl_data/generate_sft_solutions.py" \
    --base-url "http://127.0.0.1:${LB_PORT}/v1" \
    --model "${SERVED_NAME}" \
    --k "${SFT_GEN_K:-2}" \
    --concurrency "${SFT_GEN_CONCURRENCY:-18}" \
    --target-solved "${TARGET_SFT_ROWS:-500}" \
    --fewshot "${ROOT}/training/rl_data/sft_fewshot.json" \
    "${hint_args[@]}" \
    --local-only \
    "$@"
  echo "[ok] local SFT generation finished; unsolved rows in data/rl/sft_unsolved.jsonl"
else
  echo "[sft-gen] SKIP_LOCAL=1; keep existing local rows and API-fill the rest"
fi

n_now="$(wc -l < "${ROOT}/data/rl/sft_solutions.jsonl" | tr -d ' ' || echo 0)"
skip_api="${SKIP_API_FILL:-0}"
if systemctl --user is-active --quiet hybrid-grpo-pilot.service 2>/dev/null; then
  echo "[sft-gen] hybrid pilot still using the remote API; skip API fill (solved=${n_now})"
  skip_api=1
fi
if [[ "${n_now}" -ge "${TARGET_SFT_ROWS:-500}" ]]; then
  echo "[sft-gen] already have ${n_now} SFT rows (>= ${TARGET_SFT_ROWS:-500}); skip API fill"
  skip_api=1
fi

# OPENAI_* pairs in .env. SFT_API_PAIR=1 is the key on line 2. Skip exhausted tokens.
export SFT_API_MODEL="${SFT_API_MODEL:-deepseek-v4-flash}"
set +e
readarray -t _oa < <("${PYTHON}" "${ROOT}/training/rl_data/pick_sft_api.py" "${ROOT}")
_pick_rc=$?
set -e
REMOTE_BASE="${_oa[0]:-}"
REMOTE_KEY="${_oa[1]:-}"
sft_api_model="${SFT_API_MODEL:-${_oa[2]:-deepseek-v4-flash}}"
if [[ "${_pick_rc}" -ne 0 ]]; then
  echo "[sft-gen] no usable API pair (pick_rc=${_pick_rc}); skip API fill"
  skip_api=1
fi
echo "[sft-gen] using .env API pair ${_oa[3]:-?} model=${sft_api_model} hint_gold=${SFT_HINT_GOLD:-0}"

if [[ "${skip_api}" != "1" && -n "${REMOTE_KEY}" && "${REMOTE_KEY}" != "EMPTY" && -n "${REMOTE_BASE}" && "${REMOTE_BASE}" != *"127.0.0.1"* ]]; then
  echo "[sft-gen] filling unsolved via API model=${sft_api_model} base=${REMOTE_BASE}"
  hint_args=()
  if [[ "${SFT_HINT_GOLD:-1}" == "1" ]]; then
    hint_args+=(--hint-gold)
  else
    hint_args+=(--no-hint-gold)
  fi
  OPENAI_BASE_URL="${REMOTE_BASE}" OPENAI_API_KEY="${REMOTE_KEY}" SFT_API_MODEL="${sft_api_model}" \
    "${PYTHON}" "${ROOT}/training/rl_data/generate_sft_solutions.py" \
      --api-only \
      --k "${SFT_GEN_K:-2}" \
      --api-k "${SFT_GEN_K:-2}" \
      --concurrency "${SFT_API_CONCURRENCY:-8}" \
      --target-solved "${TARGET_SFT_ROWS:-500}" \
      --temperature "${SFT_API_TEMPERATURE:-0.4}" \
      --max-tokens "${SFT_API_MAX_TOKENS:-4096}" \
      --timeout "${SFT_API_TIMEOUT:-300}" \
      --fewshot "${ROOT}/training/rl_data/sft_fewshot.json" \
      "${hint_args[@]}" \
      "$@"
  if [[ "${AUDIT_SFT:-1}" == "1" ]]; then
    echo "[sft-gen] auditing SFT quality"
    "${PYTHON}" "${ROOT}/training/rl_data/audit_sft_solutions.py" \
      --input "${ROOT}/data/rl/sft_solutions.jsonl" \
      --heldout "${ROOT}/data/rl/heldout_eval.jsonl" \
      --report "${ROOT}/data/rl/sft_quality_report.json" \
      --apply || true
  fi
else
  echo "[hint] skip API fill (skip_api=${skip_api}); local solved=${n_now}"
fi

if [[ "${KEEP_LOCAL_GENERATORS:-0}" != "1" ]]; then
  for i in 0 1 2; do
    RUN_ID="${RUN_IDS[$i]}" PORT="${PORTS[$i]}" PID_FILE="${LOG_DIR}/${RUN_IDS[$i]}_vllm.pid" \
      bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop || true
  done
  if [[ -f "${LOG_DIR}/sft_gen_lb.pid" ]]; then
    old="$(cat "${LOG_DIR}/sft_gen_lb.pid" 2>/dev/null || true)"
    if [[ -n "${old}" ]]; then
      kill -TERM "${old}" 2>/dev/null || true
    fi
    rm -f "${LOG_DIR}/sft_gen_lb.pid"
  fi
fi
