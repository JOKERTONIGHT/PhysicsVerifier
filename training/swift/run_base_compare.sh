#!/usr/bin/env bash
# Zero-shot compare Qwen3-8B vs optional Intern-S1-mini / Qwen2.5-Math-7B.
# Usage: HELDOUT=data/rl/tiers/eval_easy.jsonl bash training/swift/run_base_compare.sh
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
HELDOUT="${HELDOUT:-${ROOT}/data/rl/heldout_eval_trusted.jsonl}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/base_compare}"
mkdir -p "${OUT_ROOT}"

export HELDOUT_JSONL="${HELDOUT}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"

run_one() {
  local name="$1" model="$2" thinking="${3:-0}"
  if [[ ! -f "${model}/config.json" ]]; then
    echo "[skip] ${name}: no config at ${model}"
    return 0
  fi
  echo "[eval] ${name} thinking=${thinking} ${model}"
  ENABLE_THINKING="${thinking}" N_SAMPLES="${N_SAMPLES}" MAX_TOKENS="${MAX_TOKENS}" \
    HELDOUT_JSONL="${HELDOUT}" \
    bash "${ROOT}/training/swift/eval_heldout_fast.sh" "${model}" "${OUT_ROOT}/${name}"
}

run_one "qwen3_8b_think_off" "${QWEN8B_MODEL_DIR}" 0
run_one "qwen3_8b_think_on" "${QWEN8B_MODEL_DIR}" 1
run_one "intern_s1_mini" "${INTERN_S1_MINI:-/slow_share/jinjianhan/models/Intern-S1-mini}" 0
run_one "qwen25_math_7b" "${QWEN25_MATH:-/slow_share/cyd/models/Qwen2.5-Math-7B-Instruct}" 0
echo "[ok] wrote ${OUT_ROOT}"
