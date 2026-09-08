#!/usr/bin/env bash
# Merge-or-eval each full GRPO checkpoint with the outcome-only gate:
# part_avg@4 >= 0.252 AND degrade_rate <= 0.05 AND no_boxed <= 0.05.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
CKPT="${1:-${QWEN8B_OUTCOME_CKPT}}"
BASE_MODEL="${QWEN8B_MODEL_DIR}"
BASE_SCORES="${BASE_SCORES:-${ROOT}/results/hipho_baseline_matrix_8b/base_8b_h88/heldout_scores.json}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PORT="${PORT:-8766}"
SUMMARY="${SUMMARY:-${CKPT}/gate_summary.json}"
MIN_PART="${GATE_MIN_PART:-0.252}"
MAX_DEGRADE="${GATE_MAX_DEGRADE:-0.05}"
MAX_NO_BOX="${GATE_MAX_NO_BOXED:-0.05}"
PY="${VENV_PY}"

shopt -s nullglob
ckpts=()
if [[ -d "${CKPT}" ]]; then
  for d in "${CKPT}"/v*-*/checkpoint-* "${CKPT}"/checkpoint-*; do
    [[ -d "${d}" ]] || continue
    [[ "${d}" == *-merged ]] && continue
    [[ -f "${d}/config.json" ]] || continue
    [[ -f "${d}/adapter_config.json" ]] && continue
    ckpts+=("${d}")
  done
fi
if [[ ${#ckpts[@]} -eq 0 ]]; then
  echo "[error] no full checkpoints under ${CKPT}" >&2
  exit 2
fi
mapfile -t ckpts < <(printf '%s\n' "${ckpts[@]}" | sort -V)

"${PY}" - <<'PY' "${SUMMARY}" "${CKPT}"
import json, sys
from pathlib import Path
Path(sys.argv[1]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[1]).write_text(json.dumps({"ckpt_root": sys.argv[2], "checkpoints": []}, indent=2))
PY

any_pass=0
for ckpt in "${ckpts[@]}"; do
  # Swift merge can rewrite tokenizer_config; pin the base tokenizer.
  for tok in tokenizer.json tokenizer_config.json; do
    if [[ -f "${BASE_MODEL}/${tok}" ]]; then
      cp -f "${BASE_MODEL}/${tok}" "${ckpt}/${tok}"
    fi
  done
  out="${ckpt}/heldout_fast_eval"
  if [[ -f "${out}/heldout_scores.json" && -f "${out}/gate.json" ]]; then
    echo "[gate] skip eval ${ckpt}; scores already exist"
    rc=0
  else
    set +e
    MAX_SAMPLES=0 CUDA_DEVICE="${CUDA_DEVICE}" PORT="${PORT}" \
      BASE_SCORES="${BASE_SCORES}" OUT="${out}" \
      bash "${ROOT}/training/swift/eval_heldout_fast.sh" "${ckpt}" "${out}"
    rc=$?
    set -e
  fi
  "${PY}" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "${ROOT}")
from evaluation.benchmarks.hipho.score_hipho_predictions import evaluate_gate
summary_path = Path("${SUMMARY}")
report = json.loads(summary_path.read_text()) if summary_path.is_file() else {"checkpoints": []}
scores_path = Path("${out}/heldout_scores.json")
base_path = Path("${BASE_SCORES}")
sft = json.loads(scores_path.read_text()) if scores_path.is_file() else {}
base = json.loads(base_path.read_text()) if base_path.is_file() else {}
gate = evaluate_gate(
    sft, base,
    min_part_avg=float("${MIN_PART}"),
    max_degrade_rate=float("${MAX_DEGRADE}"),
    max_no_boxed_rate=float("${MAX_NO_BOX}"),
)
gate["ckpt"] = "${ckpt}"
gate["exit_code"] = ${rc}
if not scores_path.is_file():
    gate["pass"] = False
    gate["error"] = "missing heldout_scores.json"
Path("${out}/gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
report.setdefault("checkpoints", []).append(gate)
report["any_pass"] = any(bool(c.get("pass")) for c in report["checkpoints"])
summary_path.write_text(json.dumps(report, indent=2))
print(json.dumps(gate, indent=2))
PY
  if [[ -f "${out}/gate.json" ]] && "${PY}" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("pass") else 1)' "${out}/gate.json"; then
    any_pass=1
  fi
done
echo "[gate] summary=${SUMMARY} any_pass=${any_pass}"
[[ "${any_pass}" -eq 1 ]]
