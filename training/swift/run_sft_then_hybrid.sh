#!/usr/bin/env bash
# SFT datagen + train + heldout gate, then optional Phase-3 hybrid RL.
# Does not start from the failed llm_step onset ckpt.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
SFT_DATA="${SFT_DATA:-${ROOT}/data/rl/sft_solutions.jsonl}"
SFT_CKPT="${QWEN8B_SFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft}"
PIPE_LOG="${PIPE_LOG:-${SFT_CKPT}/sft_then_hybrid.log}"
MIN_SFT_ROWS="${MIN_SFT_ROWS:-200}"
TARGET_SFT_ROWS="${TARGET_SFT_ROWS:-800}"
# shellcheck disable=SC1091
source "${ROOT}/training/swift/gpu_idle.sh"
mkdir -p "${SFT_CKPT}" "$(dirname "${PIPE_LOG}")"
log() { echo "[sft-hybrid $(date -Iseconds)] $*" | tee -a "${PIPE_LOG}"; }

n_sft() { [[ -f "${SFT_DATA}" ]] && wc -l < "${SFT_DATA}" | tr -d ' ' || echo 0; }

find_sft_ckpt() {
  if [[ -f "${SFT_CKPT}/config.json" ]]; then
    echo "${SFT_CKPT}"
    return 0
  fi
  ls -d "${SFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | tail -1 || true
}

n="$(n_sft)"
log "sft_solutions.jsonl rows=${n} (min ${MIN_SFT_ROWS}, target ${TARGET_SFT_ROWS})"
if [[ "${n}" -lt "${MIN_SFT_ROWS}" ]]; then
  log "starting SFT datagen"
  if [[ "${SFT_USE_API:-0}" == "1" ]]; then
    set -a
    # shellcheck disable=SC1091
    [[ -f "${ROOT}/.env" ]] && source "${ROOT}/.env"
    set +a
    /data1/jinjianhan/venv/openrlhf_train/bin/python "${ROOT}/training/rl_data/generate_sft_solutions.py" \
      --api-only \
      --k "${SFT_GEN_K:-4}" \
      --api-k "${SFT_GEN_K:-4}" \
      --concurrency "${SFT_API_CONCURRENCY:-8}" \
      --max-prompts "${SFT_MAX_PROMPTS:-0}" \
      --target-solved "${TARGET_SFT_ROWS}"
  else
    bash "${ROOT}/training/rl_data/launch_sft_datagen.sh"
  fi
  n="$(n_sft)"
  if [[ "${n}" -lt "${MIN_SFT_ROWS}" ]]; then
    log "ERROR SFT data still too small: ${n}"
    exit 2
  fi
fi

if [[ "${STOP_AFTER_DATAGEN:-1}" == "1" ]]; then
  log "STOP_AFTER_DATAGEN=1; labeled ${n} rows; not starting SFT"
  exit 0
fi

latest="$(find_sft_ckpt)"
if [[ -z "${latest}" || ! -f "${latest}/config.json" ]]; then
  log "starting SFT train"
  bash "${ROOT}/training/swift/run_swift_sft_8b.sh"
  sft_pid="$(cat "${SFT_CKPT}/swift_sft.pid")"
  while kill -0 "${sft_pid}" 2>/dev/null; do
    sleep 60
  done
  log "SFT process exited"
  latest="$(find_sft_ckpt)"
else
  log "SFT ckpt already present; skip train (${latest})"
fi
[[ -n "${latest}" && -f "${latest}/config.json" ]] || { log "ERROR missing SFT checkpoint"; exit 2; }

log "SFT heldout gate on ${latest}"
if [[ -z "${GATE_GPU:-}" ]]; then
  GATE_GPU="$(wait_idle_csv 1)"
  GATE_GPU="${GATE_GPU%%,*}"
fi
if ! CUDA_DEVICE="${GATE_GPU}" PORT="${GATE_PORT:-8766}" \
    QWEN8B_SFT_CKPT="${latest}" bash "${ROOT}/training/swift/run_sft_gate_eval.sh" "${latest}"; then
  log "ERROR SFT failed heldout gate; stop before RL"
  exit 3
fi

if [[ "${SKIP_PREPARE_RL:-0}" == "1" ]]; then
  log "SKIP_PREPARE_RL=1; SFT+gate done"
  exit 0
fi

log "prepare pass-rate band + hybrid GRPO"
bash "${ROOT}/training/swift/prepare_hybrid_full_rl.sh"
log "pipeline launch complete"
