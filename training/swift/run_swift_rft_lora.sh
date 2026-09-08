#!/usr/bin/env bash
# 2-GPU LoRA RFT of Qwen3-8B on on-policy correct rollouts. 1 epoch, not loss-chasing.
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
MODEL_DIR="${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
CKPT="${QWEN8B_RFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-rft-lora}"
DATA="${RFT_DATA:-${ROOT}/data/rl/rft_solutions.jsonl}"
LOG_FILE="${LOG_FILE:-${CKPT}/swift_rft_lora.log}"
NPROC="${NPROC_PER_NODE:-2}"
TRAIN_GPUS="${CUDA_VISIBLE_DEVICES:-6,7}"

[[ -x "${SWIFT_VENV}/bin/swift" ]] || { echo "[error] missing ${SWIFT_VENV}/bin/swift" >&2; exit 2; }
[[ -s "${DATA}" ]] || { echo "[error] missing RFT data ${DATA}" >&2; exit 2; }
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "[error] missing model ${MODEL_DIR}" >&2; exit 2; }
n="$(wc -l < "${DATA}")"
n="${n//[[:space:]]/}"
if [[ "${n}" -lt 20 ]]; then
  echo "[error] RFT data too small: ${n} rows (need >=20)" >&2
  exit 2
fi

# ~3 checkpoints over 1 epoch. global_batch = nproc * bs * gas
bs="${SFT_BS:-1}"
gas="${SFT_GAS:-8}"
global_bs=$(( NPROC * bs * gas ))
steps_per_epoch=$(( (n + global_bs - 1) / global_bs ))
save_steps="${SFT_SAVE_STEPS:-$(( (steps_per_epoch + 2) / 3 ))}"
if [[ "${save_steps}" -lt 1 ]]; then
  save_steps=1
fi

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

echo "[rft-lora] n_rows=${n} gpus=${TRAIN_GPUS} steps/epoch=${steps_per_epoch} save_steps=${save_steps} out=${CKPT}"
run_sft() {
  "${SWIFT_VENV}/bin/swift" sft \
    --model "${MODEL_DIR}" \
    --dataset "${DATA}" \
    --tuner_type lora \
    --lora_rank "${LORA_RANK:-32}" \
    --lora_alpha "${LORA_ALPHA:-64}" \
    --target_modules all-linear \
    --torch_dtype bfloat16 \
    --attn_impl sdpa \
    --num_train_epochs "${SFT_EPOCHS:-1}" \
    --per_device_train_batch_size "${bs}" \
    --gradient_accumulation_steps "${gas}" \
    --learning_rate "${SFT_LR:-1e-4}" \
    --max_length "${SFT_MAX_LEN:-8192}" \
    --gradient_checkpointing true \
    --deepspeed zero2 \
    --logging_steps 1 \
    --save_steps "${save_steps}" \
    --save_only_model true \
    --save_total_limit 3 \
    --eval_strategy no \
    --dataloader_num_workers 0 \
    --use_hf true \
    --output_dir "${CKPT}" \
    --logging_dir "${CKPT}/runs" \
    --report_to tensorboard
}

if [[ "${BACKGROUND:-0}" == "1" ]]; then
  nohup env \
    CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    NPROC_PER_NODE="${NPROC}" \
    MASTER_ADDR=127.0.0.1 \
    CUDA_HOME="${CUDA_HOME}" \
    PATH="${PATH}" \
    DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK}" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    TMPDIR="${TMPDIR}" TEMP="${TEMP}" TMP="${TMP}" \
    HF_DATASETS_CACHE="${HF_DATASETS_CACHE}" HF_HOME="${HF_HOME}" \
    bash -c 'run_sft' >>"${LOG_FILE}" 2>&1 &
  echo $! >"${CKPT}/swift_rft_lora.pid"
  echo "[rft-lora] pid=$(cat "${CKPT}/swift_rft_lora.pid") log=${LOG_FILE}"
else
  run_sft 2>&1 | tee -a "${LOG_FILE}"
fi
