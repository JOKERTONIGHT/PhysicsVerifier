#!/usr/bin/env bash
# Pack the outcome-only GRPO code + 1.1 MB data for another machine.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-${ROOT}/outcome_rl_bundle.tgz}"
STAGE="$(mktemp -d "${TMPDIR:-/tmp}/outcome_rl_pack.XXXXXX")"
trap 'rm -rf "${STAGE}"' EXIT
cd "${ROOT}"

paths=(
  docs/outcome_rl_runbook.md
  training/swift
  training/rl_data
  training/reward_server
  training/compat
  training/openrlhf
  training/tests
  evaluation/benchmarks/hipho
  evaluation/tests
  data/rl/swift_prompts_hybrid_band.jsonl
  data/rl/heldout_eval_trusted.jsonl
  data/rl/heldout_eval.jsonl
  results/hipho_baseline_matrix_8b/base_8b_h88/heldout_scores.json
)
missing=()
for p in "${paths[@]}"; do
  if [[ ! -e "${p}" ]]; then
    missing+=("${p}")
  fi
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "[error] missing: ${missing[*]}" >&2
  exit 2
fi

mkdir -p "${STAGE}"
tar -cf - \
  --exclude 'training/swift/train_env.sh' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "${paths[@]}" | tar -C "${STAGE}" -xf -
printf '%s\n' \
  'See docs/outcome_rl_runbook.md.' \
  'cp training/swift/train_env.example.sh training/swift/train_env.sh' \
  >"${STAGE}/README.txt"
tar -C "${STAGE}" -czf "${OUT}" .
echo "[ok] wrote ${OUT} ($(du -h "${OUT}" | awk '{print $1}'))"
