#!/usr/bin/env python3
"""Compare per-question correctness across HiPhO baseline matrix models."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.compat.math_grading import extract_answer, grade_answer_verl

LABELS = ["base_30b", "global_step5", "global_step10"]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _labels(row: Dict[str, Any]) -> List[str]:
    lab = row.get("answer") or row.get("label")
    if lab is None:
        return []
    if isinstance(lab, list):
        return [str(x) for x in lab]
    return [str(lab)]


def _correct(row: Dict[str, Any]) -> bool:
    pred = row.get("prediction", "")
    labs = _labels(row)
    return any(grade_answer_verl(pred, gt) for gt in labs) if labs else False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = {lb: _load_jsonl(args.matrix_dir / lb / "predictions.jsonl") for lb in LABELS}
    n = len(data["base_30b"])
    correct = {lb: set() for lb in LABELS}
    items = []
    for i in range(n):
        flags = {lb: _correct(data[lb][i]) for lb in LABELS}
        for lb in LABELS:
            if flags[lb]:
                correct[lb].add(i)
        meta = data["base_30b"][i].get("metadata") or {}
        items.append(
            {
                "idx": i,
                "sample_id": meta.get("sample_id") or meta.get("id"),
                "correct": flags,
                "gt": _labels(data["base_30b"][i]),
                "extracted": {
                    lb: extract_answer(data[lb][i].get("prediction", "")) or ""
                    for lb in LABELS
                },
            }
        )

    all_three = correct["base_30b"] & correct["global_step5"] & correct["global_step10"]
    report = {
        "matrix_dir": str(args.matrix_dir),
        "n_samples": n,
        "correct_counts": {lb: len(correct[lb]) for lb in LABELS},
        "all_three_correct": sorted(all_three),
        "only_base": sorted(correct["base_30b"] - correct["global_step5"] - correct["global_step10"]),
        "only_step5": sorted(correct["global_step5"] - correct["base_30b"] - correct["global_step10"]),
        "only_step10": sorted(correct["global_step10"] - correct["base_30b"] - correct["global_step5"]),
        "symmetric_diff": {
            "base_vs_step5": sorted(
                (correct["base_30b"] - correct["global_step5"]) | (correct["global_step5"] - correct["base_30b"])
            ),
            "base_vs_step10": sorted(
                (correct["base_30b"] - correct["global_step10"]) | (correct["global_step10"] - correct["base_30b"])
            ),
            "step5_vs_step10": sorted(
                (correct["global_step5"] - correct["global_step10"])
                | (correct["global_step10"] - correct["global_step5"])
            ),
        },
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "correct_counts": report["correct_counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
