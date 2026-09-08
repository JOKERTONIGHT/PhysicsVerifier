#!/usr/bin/env bash
# Phase 3: SFT actor + pass-rate band + hybrid GRPO up to 30 steps + optional onset eval.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
SFT_CKPT="${QWEN8B_SFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft}"
MODEL_DIR="${QWEN8B_MODEL_DIR:-}"
# shellcheck disable=SC1091
source "${ROOT}/training/swift/gpu_idle.sh"
if [[ -z "${MODEL_DIR}" ]]; then
  if [[ -f "${SFT_CKPT}/config.json" ]]; then
    MODEL_DIR="${SFT_CKPT}"
  else
    MODEL_DIR="$(ls -d "${SFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | tail -1 || true)"
  fi
fi
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "[error] missing SFT ckpt ${MODEL_DIR}" >&2; exit 2; }
if [[ "${MODEL_DIR}" == *"deepseek-v4-flash-grpo-onset"* ]]; then
  echo "[error] refusing failed llm_step onset ckpt" >&2
  exit 2
fi

PROMPTS="${PROMPT_DATA:-}"
if [[ -z "${PROMPTS}" ]]; then
  if [[ -s "${ROOT}/data/rl/swift_prompts_max2048_no_disagree.jsonl" ]]; then
    PROMPTS="${ROOT}/data/rl/swift_prompts_max2048_no_disagree.jsonl"
  else
    PROMPTS="${ROOT}/data/rl/swift_prompts_max2048.jsonl"
  fi
fi
BANDED="${BANDED_PROMPTS:-${ROOT}/data/rl/swift_prompts_hybrid_band.jsonl}"
ROLLOUT="${PASSRATE_ROLLOUT:-${ROOT}/data/rl/sft_passrate_rollouts.jsonl}"
PORT="${PORT:-8766}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
if [[ -z "${CUDA_DEVICE:-}" ]]; then
  CUDA_DEVICE="$(wait_idle_csv 1)"
  CUDA_DEVICE="${CUDA_DEVICE%%,*}"
fi

if [[ ! -s "${BANDED}" || "${REFRESH_BAND:-0}" == "1" ]]; then
  echo "[prep] sampling pass-rates from ${MODEL_DIR} gpu=${CUDA_DEVICE}"
  RUN_ID="hybrid_passrate" MODEL_DIR="${MODEL_DIR}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
    MAX_LEN=8192 GPU_UTIL=0.45 SERVED_NAME=qwen3-8b \
    bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" start
  "${PYTHON}" "${ROOT}/training/swift/rollout_pass_rates.py" \
    --prompts "${PROMPTS}" \
    --output "${ROLLOUT}" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model qwen3-8b \
    --n-samples "${N_SAMPLES:-8}" \
    --max-prompts "${MAX_PROMPTS:-0}"
  RUN_ID="hybrid_passrate" PORT="${PORT}" bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop || true
  "${PYTHON}" "${ROOT}/training/swift/filter_swift_by_pass_rate.py" \
    --prompts "${PROMPTS}" \
    --rollouts "${ROLLOUT}" \
    --output "${BANDED}" \
    --min-pass-rate "${MIN_PASS_RATE:-0.05}" \
    --max-pass-rate "${MAX_PASS_RATE:-0.95}"
fi
[[ -s "${BANDED}" ]] || { echo "[error] empty banded prompts ${BANDED}" >&2; exit 2; }

export MODE=full
export QWEN8B_MODEL_DIR="${MODEL_DIR}"
export PROMPT_DATA="${BANDED}"
export MAX_STEPS="${MAX_STEPS:-30}"
export NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
export MAX_COMPLETION_LEN="${MAX_COMPLETION_LEN:-3072}"
export MAX_LENGTH="${MAX_LENGTH:-5120}"
export BETA="${BETA:-0.04}"
export OVERLONG_FILTER="${OVERLONG_FILTER:-true}"
export SKIP_CALIBRATION="${SKIP_CALIBRATION:-1}"
export SKIP_SMOKE_GATE="${SKIP_SMOKE_GATE:-1}"
echo "[prep] waiting for 4 idle GPUs before full GRPO"
wait_idle_csv 4 >/dev/null
bash "${ROOT}/training/swift/launch_hybrid_grpo_4gpu.sh"
echo "[ok] full hybrid RL launched from ${MODEL_DIR} prompts=${BANDED}"

if [[ "${WAIT_AND_EVAL:-0}" == "1" ]]; then
  echo "[prep] waiting for hybrid-grpo-full.service"
  while systemctl --user is-active --quiet hybrid-grpo-full.service; do
    sleep 60
  done
  MODE=full bash "${ROOT}/training/swift/run_hybrid_onset_eval.sh"
fi
