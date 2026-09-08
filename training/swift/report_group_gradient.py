#!/usr/bin/env python3
"""Measure GRPO-usable gradient from n-sample rollouts.

A prompt is mixed if its n samples are neither all-correct nor all-wrong.
Exit 0 if mixed_rate, truncation, and boxed-rate all clear the gates.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.benchmarks.hipho.score_hipho_predictions import looks_truncated


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(
    rows: List[Dict[str, Any]],
    *,
    min_mixed: float = 0.60,
    max_trunc: float = 0.10,
    min_boxed: float = 0.95,
) -> Dict[str, Any]:
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_trunc = 0
    n_boxed = 0
    n_boxed_complete = 0
    n_complete = 0
    n_rep = 0
    for rec in rows:
        sid = str(rec.get("sample_id") or "")
        by_id[sid].append(rec)
        text = str(rec.get("response") or rec.get("prediction") or rec.get("completion") or "")
        trunc = rec.get("truncated")
        if trunc is None:
            trunc = looks_truncated(text)
        n_trunc += int(bool(trunc))
        no_box = rec.get("no_boxed")
        if no_box is None:
            no_box = "\\boxed" not in text
        boxed = not bool(no_box)
        n_boxed += int(boxed)
        if not trunc:
            n_complete += 1
            n_boxed_complete += int(boxed)
        n_rep += int(bool(rec.get("repetitive")))
    n_rows = max(len(rows), 1)
    n_ids = max(len(by_id), 1)
    mixed = 0
    all_wrong = 0
    all_right = 0
    n_acc = 0
    for recs in by_id.values():
        accs = [bool(r.get("acc")) for r in recs]
        n_acc += sum(accs)
        if accs and min(accs) is False and max(accs) is True:
            mixed += 1
        elif accs and all(accs):
            all_right += 1
        elif accs and not any(accs):
            all_wrong += 1
    mixed_rate = mixed / n_ids
    trunc_rate = n_trunc / n_rows
    boxed_rate = n_boxed / n_rows
    # GRPO --overlong_filter drops truncated rows; gate boxed on completed samples.
    boxed_rate_complete = n_boxed_complete / max(n_complete, 1)
    pass_at_1 = n_acc / n_rows
    report = {
        "n_prompts": len(by_id),
        "n_records": len(rows),
        "mixed_rate": mixed_rate,
        "all_wrong_rate": all_wrong / n_ids,
        "all_right_rate": all_right / n_ids,
        "trunc_rate": trunc_rate,
        "boxed_rate": boxed_rate,
        "boxed_rate_complete": boxed_rate_complete,
        "repetition_rate": n_rep / n_rows,
        "pass@1": pass_at_1,
        "gates": {
            "min_mixed": min_mixed,
            "max_trunc": max_trunc,
            "min_boxed": min_boxed,
            "boxed_on": "complete_nontruncated",
        },
        "pass": mixed_rate >= min_mixed and trunc_rate <= max_trunc and boxed_rate_complete >= min_boxed,
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-mixed", type=float, default=0.60)
    p.add_argument("--max-trunc", type=float, default=0.10)
    p.add_argument("--min-boxed", type=float, default=0.95)
    args = p.parse_args()
    report = summarize(
        _load_jsonl(args.rollouts),
        min_mixed=args.min_mixed,
        max_trunc=args.max_trunc,
        min_boxed=args.min_boxed,
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
