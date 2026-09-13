#!/usr/bin/env bash
# Base pass@8 on easy/mid/hard eval splits. Writes results/tier_ladder/ladder.json.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
MODEL="${1:-${QWEN8B_MODEL_DIR}}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/results/tier_ladder}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
mkdir -p "${OUT_ROOT}"

eval_tier() {
  local tier="$1"
  local src="${ROOT}/data/rl/tiers/eval_${tier}.jsonl"
  if [[ ! -s "${src}" ]]; then
    echo "[skip] missing ${src}"
    return 0
  fi
  HELDOUT_JSONL="${src}" N_SAMPLES="${N_SAMPLES}" MAX_TOKENS="${MAX_TOKENS}" \
    bash "${ROOT}/training/swift/eval_heldout_fast.sh" "${MODEL}" "${OUT_ROOT}/${tier}"
}

eval_tier easy
eval_tier mid
if [[ -s "${ROOT}/data/rl/tiers/eval_hard.jsonl" ]]; then
  eval_tier hard
fi
"${VENV_PY}" "${ROOT}/training/swift/build_difficulty_ladder.py" \
  --easy "${OUT_ROOT}/easy/heldout_scores.json" \
  --mid "${OUT_ROOT}/mid/heldout_scores.json" \
  --hard "${OUT_ROOT}/hard/heldout_scores.json" \
  --output "${OUT_ROOT}/ladder.json"
