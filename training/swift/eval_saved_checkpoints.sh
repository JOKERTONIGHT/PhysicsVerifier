#!/usr/bin/env bash
# Score every GRPO checkpoint on the same-distribution val set (k=8).
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_load_train_env.sh"
ROOT="${PHYSICS_ROOT}"
CKPT="${1:?usage: eval_saved_checkpoints.sh CKPT_DIR}"
HELDOUT_JSONL="${HELDOUT_JSONL:-${ROOT}/data/rl/val_same_dist.jsonl}"
OUT_ROOT="${OUT_ROOT:-${CKPT}/val_eval}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
export OUT_ROOT
mkdir -p "${OUT_ROOT}"
shopt -s nullglob
ckpts=("${CKPT}"/checkpoint-* "${CKPT}"/v*-*/checkpoint-*)
if (( ${#ckpts[@]} == 0 )); then
  echo "[error] no checkpoints under ${CKPT}" >&2
  exit 2
fi
for dir in "${ckpts[@]}"; do
  [[ -f "${dir}/config.json" ]] || continue
  name="$(basename "${dir}")"
  echo "[eval] ${name}"
  HELDOUT_JSONL="${HELDOUT_JSONL}" N_SAMPLES="${N_SAMPLES}" MAX_TOKENS="${MAX_TOKENS}" \
    bash "${ROOT}/training/swift/eval_heldout_fast.sh" "${dir}" "${OUT_ROOT}/${name}"
done
"${VENV_PY}" - <<'PY'
import json, os, re
from pathlib import Path
root = Path(os.environ["OUT_ROOT"])
rows = []
for scores in sorted(root.glob("checkpoint-*/heldout_scores.json"), key=lambda p: p.stat().st_mtime):
    blob = json.loads(scores.read_text())
    step = int(re.search(r"checkpoint-(\d+)", str(scores)).group(1))
    rows.append({
        "step": step,
        "part_avg_at_k": blob.get("part_avg_at_k"),
        "part_avg_at_k_ci": blob.get("part_avg_at_k_ci"),
        "part_pass_minus_avg": blob.get("part_pass_minus_avg"),
        "n_samples": blob.get("n_samples"),
        "k": blob.get("k"),
        "path": str(scores),
    })
rows.sort(key=lambda r: r["step"])
out = root / "val_curve.json"
payload = {"n": len(rows), "rows": rows}
out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
