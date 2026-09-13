#!/usr/bin/env python3
"""Measure completion lengths from predictions jsonl (chars and boxed rate)."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.compat.part_scoring import extract_all_boxed, looks_truncated


def _load(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    i = min(len(ys) - 1, max(0, int(round(q * (len(ys) - 1)))))
    return float(ys[i])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--budget-tokens", type=int, default=4096)
    p.add_argument("--chars-per-token", type=float, default=3.3)
    args = p.parse_args()
    rows = _load(args.predictions)
    lengths = []
    n_box = 0
    n_trunc = 0
    over_budget = 0
    budget_chars = args.budget_tokens * args.chars_per_token
    for row in rows:
        text = str(row.get("prediction") or row.get("response") or row.get("completion") or "")
        lengths.append(float(len(text)))
        if extract_all_boxed(text):
            n_box += 1
        if looks_truncated(text):
            n_trunc += 1
        if len(text) >= budget_chars:
            over_budget += 1
    n = max(len(lengths), 1)
    report = {
        "n": len(lengths),
        "p50_chars": percentile(lengths, 0.5),
        "p90_chars": percentile(lengths, 0.9),
        "p95_chars": percentile(lengths, 0.95),
        "max_chars": max(lengths) if lengths else 0.0,
        "mean_chars": statistics.mean(lengths) if lengths else 0.0,
        "boxed_rate": n_box / n,
        "truncated_rate": n_trunc / n,
        "over_budget_rate": over_budget / n,
        "budget_tokens": args.budget_tokens,
        "recommend_ok": (over_budget / n) < 0.10,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["recommend_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
