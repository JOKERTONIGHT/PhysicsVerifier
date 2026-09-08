#!/usr/bin/env bash
# Onset-style heldout exact + HiPhO-TO MNS for hybrid GRPO. Does not use Internal150.
set -euo pipefail
ROOT="${PHYSICS_ROOT:-/home/jinjianhan/PhysicsVerifier}"
MODE="${MODE:-full}"
if [[ "${MODE}" == "pilot" ]]; then
  export QWEN8B_LLM_VERIFIER_CKPT="${QWEN8B_HYBRID_PILOT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-pilot10}"
  export BASE_MODEL="${QWEN8B_MODEL_DIR:-/slow_share/jinjianhan/models/Qwen3-8B}"
  export EVAL_STEPS="${EVAL_STEPS:-0 5 10}"
else
  export QWEN8B_LLM_VERIFIER_CKPT="${QWEN8B_HYBRID_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-rl}"
  SFT_CKPT="${QWEN8B_SFT_CKPT:-/slow_share/jinjianhan/ckpt/qwen3-8b-physics-sft}"
  if [[ -z "${QWEN8B_MODEL_DIR:-}" ]]; then
    if [[ -f "${SFT_CKPT}/config.json" ]]; then
      export BASE_MODEL="${SFT_CKPT}"
    else
      export BASE_MODEL="$(ls -d "${SFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | tail -1 || true)"
    fi
  else
    export BASE_MODEL="${QWEN8B_MODEL_DIR}"
  fi
  [[ -f "${BASE_MODEL}/config.json" ]] || export BASE_MODEL="/slow_share/jinjianhan/models/Qwen3-8B"
  export EVAL_STEPS="${EVAL_STEPS:-0 10 20 30}"
fi
export OUT_DIR="${OUT_DIR:-${QWEN8B_LLM_VERIFIER_CKPT}/onset_eval}"
if [[ "${QWEN8B_LLM_VERIFIER_CKPT}" == *"deepseek-v4-flash-grpo-onset"* ]]; then
  echo "[error] refusing failed llm_step onset ckpt" >&2
  exit 2
fi
bash "${ROOT}/training/swift/run_llm_verifier_onset_eval.sh"
