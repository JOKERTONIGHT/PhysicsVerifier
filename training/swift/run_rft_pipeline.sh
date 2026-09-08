#!/usr/bin/env bash
# Sequential RFT pipeline: rebase baseline → pass-rate → build RFT → LoRA → gate → GRPO wait.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
PYTHON="${PYTHON:-/data1/jinjianhan/venv/openrlhf_train/bin/python}"
VENV="${VENV:-${ROOT}/.venv}"
BASE_MODEL="${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
BASE_OUT="${ROOT}/results/hipho_baseline_matrix_8b/base_8b_h88"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
mkdir -p "${LOG_DIR}" "${BASE_OUT}"
# shellcheck disable=SC1091
source "${ROOT}/training/swift/gpu_idle.sh"

CUDA_DEVICE="${CUDA_DEVICE:-6}"
PORT="${PORT:-8766}"

if [[ "${SKIP_BASELINE:-0}" != "1" ]]; then
  echo "[pipeline] rebase base heldout avg@4 gpu=${CUDA_DEVICE}"
  MAX_SAMPLES=0 N_SAMPLES=4 TEMPERATURE=0.6 MAX_TOKENS=8192 MAX_LEN=16384 GPU_UTIL=0.85 \
    CUDA_DEVICE="${CUDA_DEVICE}" PORT="${PORT}" \
    bash "${ROOT}/training/swift/eval_heldout_fast.sh" "${BASE_MODEL}" "${BASE_OUT}"
fi

echo "[pipeline] pass-rate n=8 on 554 SFT golds"
RUN_ID="${VLLM_RUN_ID:-rft_passrate}" MODEL_DIR="${BASE_MODEL}" PORT="${PORT}" CUDA_DEVICE="${CUDA_DEVICE}" \
  MAX_LEN=16384 GPU_UTIL=0.85 SERVED_NAME=qwen3-8b \
  bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" start
"${PYTHON}" "${ROOT}/training/swift/rollout_pass_rates.py" \
  --prompts "${ROOT}/data/rl/sft_solutions.jsonl" \
  --output "${ROOT}/data/rl/base_pass_rates_sft554.jsonl" \
  --summary "${ROOT}/data/rl/base_pass_rates_sft554_summary.json" \
  --base-url "http://127.0.0.1:${PORT}/v1" \
  --model qwen3-8b \
  --n-samples 8 \
  --temperature 0.8 \
  --max-tokens 8192 \
  --concurrency "${CONCURRENCY:-16}"

n_pass="$(python3 -c 'import json; print(int(json.load(open("'"${ROOT}"'/data/rl/base_pass_rates_sft554_summary.json")).get("n_pass_at_k") or 0))')"
echo "[pipeline] pass@8 coverage=${n_pass}"
if [[ "${n_pass}" -lt 100 ]]; then
  echo "[pipeline] expanding pass-rate pool to rl_prompts.jsonl"
  "${PYTHON}" "${ROOT}/training/swift/rollout_pass_rates.py" \
    --prompts "${ROOT}/data/rl/rl_prompts.jsonl" \
    --output "${ROOT}/data/rl/base_pass_rates_rl1235.jsonl" \
    --summary "${ROOT}/data/rl/base_pass_rates_rl1235_summary.json" \
    --base-url "http://127.0.0.1:${PORT}/v1" \
    --model qwen3-8b \
    --n-samples 8 \
    --temperature 0.8 \
    --max-tokens 8192 \
    --concurrency "${CONCURRENCY:-16}"
  cat "${ROOT}/data/rl/base_pass_rates_sft554.jsonl" "${ROOT}/data/rl/base_pass_rates_rl1235.jsonl" \
    > "${ROOT}/data/rl/base_pass_rates_combined.jsonl"
  ROLLOUTS="${ROOT}/data/rl/base_pass_rates_combined.jsonl"
  PROMPTS_FOR_RFT="${ROOT}/data/rl/rl_prompts.jsonl"
else
  ROLLOUTS="${ROOT}/data/rl/base_pass_rates_sft554.jsonl"
  PROMPTS_FOR_RFT="${ROOT}/data/rl/sft_solutions.jsonl"
fi
RUN_ID="${VLLM_RUN_ID:-rft_passrate}" PORT="${PORT}" bash "${ROOT}/evaluation/benchmarks/hipho/manage_eval_vllm.sh" stop || true

echo "[pipeline] build RFT from correct rollouts"
"${VENV}/bin/python" "${ROOT}/training/rl_data/build_rft_from_rollouts.py" \
  --rollouts "${ROLLOUTS}" \
  --prompts "${PROMPTS_FOR_RFT}" \
  --output "${ROOT}/data/rl/rft_solutions.jsonl" \
  --audit "${ROOT}/data/rl/rft_build_report.json"

echo "[pipeline] LoRA RFT: prefer 2 idle GPUs, else 1"
if RFT_GPUS="$(probe_idle_csv 2 2>/dev/null)"; then
  nproc=2
else
  RFT_GPUS="$(wait_idle_csv 1)"
  nproc=1
fi
echo "[pipeline] LoRA RFT on ${RFT_GPUS} nproc=${nproc}"
CUDA_VISIBLE_DEVICES="${RFT_GPUS}" NPROC_PER_NODE="${nproc}" \
  bash "${ROOT}/training/swift/run_swift_rft_lora.sh"

echo "[pipeline] gate-check LoRA checkpoints"
if ! CUDA_DEVICE="${CUDA_DEVICE}" PORT="${PORT}" \
  bash "${ROOT}/training/swift/run_rft_gate_check.sh"; then
  echo "[pipeline] LoRA gate did not pass; GRPO waiter will use base model"
fi

echo "[pipeline] launch GRPO waiter (needs 4 idle GPUs)"
nohup bash "${ROOT}/training/swift/wait_and_launch_hybrid_grpo.sh" \
  >>"${LOG_DIR}/wait_and_launch_hybrid_grpo.log" 2>&1 &
echo $! >"${LOG_DIR}/wait_and_launch_hybrid_grpo.pid"
echo "[pipeline] done launching waiter pid=$(cat "${LOG_DIR}/wait_and_launch_hybrid_grpo.pid")"
