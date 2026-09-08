#!/usr/bin/env bash
# Merge LoRA checkpoints and run the heldout gate on each. Writes gate_summary.json.
set -euo pipefail
# shellcheck disable=SC1091
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
WORKSPACE="${WORKSPACE_ROOT}"
RFT_CKPT="${1:-${QWEN8B_RFT_CKPT}}"
BASE_MODEL="${QWEN8B_MODEL_DIR}"
BASE_SCORES="${BASE_SCORES:-${ROOT}/results/hipho_baseline_matrix_8b/base_8b_h88/heldout_scores.json}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
PORT="${PORT:-8766}"
SUMMARY="${SUMMARY:-${RFT_CKPT}/gate_summary.json}"
mkdir -p "${ROOT}/logs"
echo $$ >"${ROOT}/logs/rft_gate_check.pid"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
export DS_SKIP_CUDA_CHECK="${DS_SKIP_CUDA_CHECK:-1}"

shopt -s nullglob
ckpts=()
if [[ -d "${RFT_CKPT}" ]]; then
  for d in "${RFT_CKPT}"/v*-*/checkpoint-* "${RFT_CKPT}"/checkpoint-*; do
    [[ -d "${d}" ]] || continue
    [[ "${d}" == *-merged ]] && continue
    [[ -f "${d}/adapter_config.json" ]] || continue
    ckpts+=("${d}")
  done
fi
if [[ ${#ckpts[@]} -eq 0 ]]; then
  echo "[error] no LoRA checkpoints under ${RFT_CKPT}" >&2
  exit 2
fi
mapfile -t ckpts < <(printf '%s\n' "${ckpts[@]}" | sort -V)

any_pass=0
"${VENV_PY}" - <<'PY' "${SUMMARY}" "${RFT_CKPT}"
import json, sys
from pathlib import Path
Path(sys.argv[1]).write_text(json.dumps({"ckpt_root": sys.argv[2], "checkpoints": []}, indent=2))
PY

for ckpt in "${ckpts[@]}"; do
  merged="${ckpt}-merged"
  if [[ ! -f "${merged}/config.json" ]]; then
    rm -rf "${merged}"
    echo "[gate] merging LoRA ${ckpt} -> ${merged}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${SWIFT_VENV}/bin/swift" export \
      --model "${BASE_MODEL}" \
      --adapters "${ckpt}" \
      --merge_lora true \
      --output_dir "${merged}"
  fi
  [[ -f "${merged}/config.json" ]] || { echo "[error] merge failed ${ckpt}" >&2; continue; }
  # Replace Swift's tokenizer_config (list extra_special_tokens) with the base files.
  for tok in tokenizer.json tokenizer_config.json; do
    if [[ -f "${BASE_MODEL}/${tok}" ]]; then
      cp -f "${BASE_MODEL}/${tok}" "${merged}/${tok}"
    fi
  done
  out="${merged}/heldout_fast_eval"
  if [[ -f "${out}/heldout_scores.json" && -f "${out}/gate.json" ]]; then
    echo "[gate] skip eval ${ckpt}; scores already exist"
    rc=0
  else
    set +e
    MAX_SAMPLES=0 CUDA_DEVICE="${CUDA_DEVICE}" PORT="${PORT}" \
      BASE_SCORES="${BASE_SCORES}" OUT="${out}" \
      bash "${ROOT}/training/swift/run_sft_gate_eval.sh" "${merged}"
    rc=$?
    set -e
  fi
  "${VENV_PY}" - <<PY
import json
from pathlib import Path
summary_path = Path("${SUMMARY}")
report = json.loads(summary_path.read_text()) if summary_path.is_file() else {"checkpoints": []}
gate_path = Path("${out}/gate.json")
gate = json.loads(gate_path.read_text()) if gate_path.is_file() else {"pass": False, "error": "missing gate.json"}
gate["ckpt"] = "${ckpt}"
gate["merged"] = "${merged}"
gate["exit_code"] = ${rc}
report.setdefault("checkpoints", []).append(gate)
report["any_pass"] = any(bool(c.get("pass")) for c in report["checkpoints"])
summary_path.write_text(json.dumps(report, indent=2))
import sys
sys.path.insert(0, "${ROOT}")
from training.swift.summarize_rft_diag import summarize_gate
collapse = summarize_gate(report)
Path("${SUMMARY}").with_name("collapse_report.json").write_text(json.dumps(collapse, indent=2), encoding="utf-8")
print(json.dumps(gate, indent=2))
print(json.dumps(collapse, indent=2), flush=True)
PY
  if [[ "${rc}" -eq 0 ]]; then
    any_pass=1
  fi
done

"${VENV_PY}" - <<PY
import json, sys
from pathlib import Path
sys.path.insert(0, "${ROOT}")
from training.swift.summarize_rft_diag import summarize_gate
report = json.loads(Path("${SUMMARY}").read_text())
collapse = summarize_gate(report)
collapse_path = Path("${SUMMARY}").with_name("collapse_report.json")
collapse_path.write_text(json.dumps(collapse, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
print(json.dumps(collapse, indent=2))
sys.exit(0 if report.get("any_pass") else 3)
PY
