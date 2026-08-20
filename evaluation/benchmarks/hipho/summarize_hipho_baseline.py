#!/usr/bin/env python3
"""Summarize HiPhO baseline matrix and compare against the base model."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wilson_ci(hits: int, n: int, z: float = 1.96) -> List[float]:
    if n <= 0:
        return [0.0, 0.0]
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _prediction_stats(pred_path: Path) -> Dict[str, float]:
    rows = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    nonempty = sum(1 for r in rows if str(r.get("prediction") or "").strip())
    lengths = [len(str(r.get("prediction") or "")) for r in rows]
    return {
        "n_samples": len(rows),
        "nonempty_rate": nonempty / max(len(rows), 1),
        "avg_prediction_chars": sum(lengths) / max(len(lengths), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, required=True)
    parser.add_argument("--base-label", default="base_30b")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    entries: List[Dict[str, Any]] = []
    for score_path in sorted(args.matrix_dir.glob("*/scores.json")):
        label = score_path.parent.name
        score = _load_json(score_path)
        pred_path = score_path.parent / "predictions.jsonl"
        pred_stats = _prediction_stats(pred_path) if pred_path.is_file() else {}
        hits = int(round(score.get("answer_acc", 0.0) * score.get("n_samples", 0)))
        ci = _wilson_ci(hits, int(score.get("n_samples", 0)))
        entries.append(
            {
                "label": label,
                "answer_acc": score.get("answer_acc"),
                "avg_process_errors": score.get("avg_process_errors"),
                "n_samples": score.get("n_samples"),
                "answer_acc_ci95": ci,
                "per_exam": score.get("per_exam", {}),
                "predictions": str(pred_path),
                **pred_stats,
            }
        )

    base = next((e for e in entries if e["label"] == args.base_label), None)
    comparisons = []
    for e in entries:
        if base is None or e["label"] == args.base_label:
            continue
        comparisons.append(
            {
                "label": e["label"],
                "delta_answer_acc": (e.get("answer_acc") or 0.0) - (base.get("answer_acc") or 0.0),
                "delta_avg_process_errors": (e.get("avg_process_errors") or 0.0) - (base.get("avg_process_errors") or 0.0),
            }
        )

    report = {
        "matrix_dir": str(args.matrix_dir),
        "base_label": args.base_label,
        "entries": entries,
        "comparisons_vs_base": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
