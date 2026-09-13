#!/usr/bin/env bash
# Wait for 4 idle GPUs, run 2-step throughput probe, then 400-step outcome-only GRPO.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
source "${PHYSICS_ROOT}/training/swift/gpu_idle.sh"
ROOT="${PHYSICS_ROOT}"
N_GPUS="${N_TRAIN_GPUS:-4}"
export WAIT_GPU_DEADLINE_SECS="${WAIT_GPU_DEADLINE_SECS:-28800}"
LOG="${ROOT}/logs/overhaul_pipeline.log"
mkdir -p "${ROOT}/logs"

{
  echo "[pipeline] $(date -u +%Y-%m-%dT%H:%M:%SZ) waiting for ${N_GPUS} idle GPUs deadline=${WAIT_GPU_DEADLINE_SECS}s"
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    if ! ids="$(wait_idle_csv "${N_GPUS}")"; then
      echo "[error] GPU wait failed; see logs/gpu_wait_failed.json"
      exit 2
    fi
    export CUDA_VISIBLE_DEVICES="${ids}"
  fi
  echo "[pipeline] gpus=${CUDA_VISIBLE_DEVICES} running difficulty ladder then 2-step probe"
  bash "${ROOT}/training/swift/run_tier_ladder.sh" || echo "[warn] ladder failed; continuing"
  bash "${ROOT}/training/swift/run_throughput_probe.sh"
  CKPT="${QWEN8B_OUTCOME_SMOKE_CKPT}"
  DEADLINE=2400
  start="$(date +%s)"
  while true; do
    if grep -qE 'swift exited status=' "${CKPT}/swift_grpo.log" 2>/dev/null; then
      break
    fi
    if (( "$(date +%s)" - start >= DEADLINE )); then
      echo "[error] probe did not finish in ${DEADLINE}s"
      exit 2
    fi
    sleep 20
  done
  echo "[pipeline] probe finished; launching 400-step outcome_only"
  export MODE=full
  export SKIP_SMOKE_GATE=1
  if [[ -s "${ROOT}/data/rl/tiers/easy_prompts.jsonl" ]]; then
    export PROMPT_DATA="${ROOT}/data/rl/tiers/easy_prompts.jsonl"
  elif [[ -s "${ROOT}/data/rl/tiers/mid_prompts.jsonl" ]]; then
    export PROMPT_DATA="${ROOT}/data/rl/tiers/mid_prompts.jsonl"
  fi
  bash "${ROOT}/training/swift/run_outcome_grpo_full.sh"
  echo "[pipeline] full GRPO launched ckpt=${QWEN8B_OUTCOME_CKPT}"
} >>"${LOG}" 2>&1
