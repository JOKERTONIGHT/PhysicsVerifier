#!/usr/bin/env bash
# Heldout gate: part-level avg@k non-degradation + degrade_rate must not rise.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
SFT_CKPT="${1:-${QWEN8B_SFT_CKPT}}"
if [[ ! -f "${SFT_CKPT}/config.json" ]]; then
  SFT_CKPT="$(ls -d "${SFT_CKPT}"/v*-*/checkpoint-* 2>/dev/null | tail -1 || true)"
fi
[[ -f "${SFT_CKPT}/config.json" ]] || { echo "[error] missing SFT ckpt" >&2; exit 2; }
BASE_SCORES="${BASE_SCORES:-${ROOT}/results/hipho_baseline_matrix_8b/base_8b_h88/heldout_scores.json}"
OUT="${OUT:-${SFT_CKPT}/heldout_fast_eval}"
set +e
MAX_SAMPLES="${MAX_SAMPLES:-0}" N_SAMPLES="${N_SAMPLES:-4}" TEMPERATURE="${TEMPERATURE:-0.6}" \
  MAX_TOKENS="${MAX_TOKENS:-8192}" CUDA_DEVICE="${CUDA_DEVICE:-0}" PORT="${PORT:-8766}" \
  bash "${ROOT}/training/swift/eval_heldout_fast.sh" "${SFT_CKPT}" "${OUT}"
eval_rc=$?
set -e
if [[ ! -f "${OUT}/heldout_scores.json" ]]; then
  echo "[error] missing ${OUT}/heldout_scores.json (eval_heldout_fast rc=${eval_rc})" >&2
  exit "${eval_rc:-2}"
fi
"${VENV_PY}" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "${ROOT}")
from evaluation.benchmarks.hipho.score_hipho_predictions import evaluate_gate
sft = json.loads(Path("${OUT}/heldout_scores.json").read_text())
base_path = Path("${BASE_SCORES}")
base = json.loads(base_path.read_text()) if base_path.is_file() else {}
report = evaluate_gate(sft, base)
Path("${OUT}/gate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
sys.exit(0 if report["pass"] else 3)
PY
