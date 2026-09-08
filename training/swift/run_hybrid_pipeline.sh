#!/usr/bin/env bash
# Background orchestrator: SFT (on idle GPUs) → wait for 10-step pilot → observe → full hybrid RL.
# Does not resume qwen3-8b-deepseek-v4-flash-grpo-onset. Train reward is not a success metric.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
export PHYSICS_ROOT="${ROOT}"
SFT_CKPT="${QWEN8B_SFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft}"
FULL_CKPT="${QWEN8B_HYBRID_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-rl}"
PIPE_LOG="${PIPE_LOG:-${FULL_CKPT}/hybrid_pipeline.log}"
mkdir -p "${SFT_CKPT}" "${FULL_CKPT}" "$(dirname "${PIPE_LOG}")"
# shellcheck disable=SC1091
source "${ROOT}/training/swift/gpu_idle.sh"
log() { echo "[hybrid-pipe $(date -Iseconds)] $*" | tee -a "${PIPE_LOG}"; }

wait_unit_idle() {
  local unit="$1"
  if ! systemctl --user is-active --quiet "${unit}"; then
    return 0
  fi
  log "waiting for ${unit} to finish"
  while systemctl --user is-active --quiet "${unit}"; do
    sleep 60
  done
  log "${unit} inactive"
}

log "start SKIP_PILOT_WAIT=${SKIP_PILOT_WAIT:-1} SKIP_OBSERVE=${SKIP_OBSERVE:-0} SKIP_SFT=${SKIP_SFT:-0} SKIP_RL=${SKIP_RL:-0}"

if [[ "${SKIP_SFT:-0}" != "1" ]]; then
  log "Phase 2: SFT datagen only (STOP_AFTER_DATAGEN=${STOP_AFTER_DATAGEN:-1})"
  STOP_AFTER_DATAGEN="${STOP_AFTER_DATAGEN:-1}" SKIP_PREPARE_RL=1 bash "${ROOT}/training/swift/run_sft_then_hybrid.sh"
  log "datagen finished rows=$(wc -l < "${ROOT}/data/rl/sft_solutions.jsonl" | tr -d ' ')"
else
  log "skip SFT"
fi

if [[ "${STOP_AFTER_DATAGEN:-1}" == "1" ]]; then
  log "STOP_AFTER_DATAGEN=1; not starting SFT train, observe, or RL"
  exit 0
fi

if [[ "${SKIP_PILOT_WAIT:-1}" != "1" ]] || systemctl --user is-active --quiet hybrid-grpo-pilot.service; then
  wait_unit_idle hybrid-grpo-pilot.service
fi

if [[ "${SKIP_OBSERVE:-0}" != "1" ]]; then
  log "Phase 1 observe (boxed / mixed-acc / clip / heldout-50)"
  if ! bash "${ROOT}/training/swift/run_hybrid_pilot_observe.sh"; then
    log "WARN observe failed or no ckpt; continuing if SFT gate passed"
  fi
fi

if [[ "${SKIP_RL:-0}" == "1" ]]; then
  log "SKIP_RL=1; stop after SFT/observe"
  exit 0
fi

log "Phase 3: pass-rate band + hybrid GRPO 30 steps from SFT"
WAIT_AND_EVAL="${WAIT_AND_EVAL:-1}" bash "${ROOT}/training/swift/prepare_hybrid_full_rl.sh"
log "pipeline complete"
