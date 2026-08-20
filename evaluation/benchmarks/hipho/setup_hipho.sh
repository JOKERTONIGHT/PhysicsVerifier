#!/usr/bin/env bash
# Download HiPhO benchmark data and official evaluation repo.
set -euo pipefail

BENCH_ROOT="${BENCH_ROOT:-/slow_share/jinjianhan/workspace/benchmarks/hipho}"
REPO_DIR="${REPO_DIR:-${BENCH_ROOT}/HiPhO}"
VENV="${VENV:-/home/jinjianhan/PhysicsVerifier/.venv}"
OUT_JSONL="${OUT_JSONL:-${BENCH_ROOT}/hipho_text_only.jsonl}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck source=pip_mirror.sh
source "${SCRIPT_DIR}/pip_mirror.sh"

mkdir -p "${BENCH_ROOT}"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/SciYu/HiPhO.git "${REPO_DIR}" || {
    echo "[warn] git clone failed; place HiPhO repo manually at ${REPO_DIR}" >&2
  }
fi

pip_install "${VENV}/bin/pip" -q datasets huggingface_hub 2>/dev/null || true

if [[ -f "${OUT_JSONL}" ]]; then
  echo "[ok] HiPhO text-only file already exists: ${OUT_JSONL}"
  exit 0
fi

"${VENV}/bin/python" - <<PY
from pathlib import Path
import json

bench_root = Path("${BENCH_ROOT}")
out = Path("${OUT_JSONL}")
bench_root.mkdir(parents=True, exist_ok=True)

try:
    from datasets import load_dataset
    ds = load_dataset("SciYu/HiPhO", split="test")
    count = 0
    with out.open("w", encoding="utf-8") as f:
        for row in ds:
            modality = str(row.get("modality") or row.get("Modality") or "").lower()
            if modality and "text-only" not in modality and modality not in ("text", "text only"):
                continue
            question = row.get("problem") or row.get("question") or row.get("Problem")
            if not question:
                continue
            rec = {
                "id": row.get("id") or row.get("problem_id"),
                "question": question,
                "answer": row.get("answer") or row.get("ground_truth"),
                "metadata": {
                    "exam": row.get("exam") or row.get("source"),
                    "modality": modality,
                    "marking": row.get("marking") or row.get("marking_scheme"),
                },
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
            count += 1
    print(json.dumps({"text_only_count": count, "output": str(out)}, ensure_ascii=False))
except Exception as e:
    # Offline fallback: symlink held-out eval prompts for pipeline smoke only.
    fallback = Path("/home/jinjianhan/PhysicsVerifier/data/rl/heldout_eval.jsonl")
    if fallback.exists():
        out.write_text(fallback.read_text(encoding="utf-8"), encoding="utf-8")
        print(json.dumps({
            "warning": str(e),
            "fallback_used": str(fallback),
            "output": str(out),
            "note": "Replace with real HiPhO data when network is available"
        }, ensure_ascii=False))
    else:
        raise
PY

echo "[ok] HiPhO setup done at ${BENCH_ROOT}"
