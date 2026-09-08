#!/usr/bin/env bash
# Diagnostic LoRA: 1 unique correct rollout per id, rank 16, lr 2e-5, save every 20 steps.
set -euo pipefail
ulimit -f unlimited 2>/dev/null || true
SLOW_TMP_ROOT="${SLOW_TMP_ROOT:-/slow_share/jinjianhan/tmp}"
export TMPDIR="${TMPDIR:-${SLOW_TMP_ROOT}/swift}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}"

ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
WORKSPACE="${WORKSPACE_ROOT:-/slow_share/jinjianhan/workspace}"
SWIFT_VENV="${SWIFT_VENV:-/data1/jinjianhan/venv/swift_train}"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"
MODEL_DIR="${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
CKPT="${QWEN8B_RFT_DIAG_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-rft-lora-diag}"
ROLLOUTS="${PASSRATE_ROLLOUT:-${ROOT}/data/rl/base_pass_rates_sft554.jsonl}"
PROMPTS="${RFT_PROMPTS:-${ROOT}/data/rl/sft_solutions.jsonl}"
DATA="${RFT_DATA:-${ROOT}/data/rl/rft_solutions_dedup.jsonl}"
LOG_FILE="${LOG_FILE:-${CKPT}/swift_rft_lora_diag.log}"
NPROC="${NPROC_PER_NODE:-1}"
TRAIN_GPUS="${CUDA_VISIBLE_DEVICES:-7}"

[[ -x "${SWIFT_VENV}/bin/swift" ]] || { echo "[error] missing ${SWIFT_VENV}/bin/swift" >&2; exit 2; }
[[ -s "${ROLLOUTS}" ]] || { echo "[error] missing rollouts ${ROLLOUTS}" >&2; exit 2; }
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "[error] missing model ${MODEL_DIR}" >&2; exit 2; }

"${PYTHON}" "${ROOT}/training/rl_data/build_rft_from_rollouts.py" \
  --rollouts "${ROLLOUTS}" \
  --prompts "${PROMPTS}" \
  --output "${DATA}" \
  --audit "${ROOT}/data/rl/rft_dedup_report.json" \
  --max-per-id 1 \
  --target-len 4600

n="$(wc -l < "${DATA}")"
n="${n//[[:space:]]/}"
if [[ "${n}" -lt 20 ]]; then
  echo "[error] dedup RFT data too small: ${n} rows" >&2
  exit 2
fi

bs="${SFT_BS:-1}"
gas="${SFT_GAS:-4}"
global_bs=$(( NPROC * bs * gas ))
steps_per_epoch=$(( (n + global_bs - 1) / global_bs ))
save_steps="${SFT_SAVE_STEPS:-20}"
epochs="${SFT_EPOCHS:-2}"

mkdir -p "${CKPT}" "${TMPDIR}/hf_datasets" "${TMPDIR}/hf_home"
export CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}"
export NPROC_PER_NODE="${NPROC}"
export MASTER_ADDR=127.0.0.1
export CUDA_HOME="${CUDA_HOME:-${WORKSPACE}/openrlhf_rl/cuda_stub}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${TMPDIR}/hf_datasets}"
export HF_HOME="${HF_HOME:-${TMPDIR}/hf_home}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[rft-diag] n_rows=${n} gpus=${TRAIN_GPUS} steps/epoch=${steps_per_epoch} save_steps=${save_steps} rank=16 lr=2e-5 out=${CKPT}"
"${SWIFT_VENV}/bin/swift" sft \
  --model "${MODEL_DIR}" \
  --dataset "${DATA}" \
  --tuner_type lora \
  --lora_rank "${LORA_RANK:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --target_modules all-linear \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --num_train_epochs "${epochs}" \
  --per_device_train_batch_size "${bs}" \
  --gradient_accumulation_steps "${gas}" \
  --learning_rate "${SFT_LR:-2e-5}" \
  --max_length "${SFT_MAX_LEN:-8192}" \
  --gradient_checkpointing true \
  --deepspeed zero2 \
  --logging_steps 1 \
  --save_steps "${save_steps}" \
  --save_only_model true \
  --save_total_limit 8 \
  --eval_strategy no \
  --dataloader_num_workers 0 \
  --use_hf true \
  --output_dir "${CKPT}" \
  --logging_dir "${CKPT}/runs" \
  --report_to tensorboard \
  2>&1 | tee -a "${LOG_FILE}"

echo "[rft-diag] training done; run gate with CUDA_DEVICE=${TRAIN_GPUS%%,*} QWEN8B_RFT_CKPT=${CKPT} bash training/swift/run_rft_gate_check.sh"
