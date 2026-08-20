#!/usr/bin/env python3
"""Summarize scale_1500 A/B matrix dual-eval outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    root = Path(args.matrix_root)
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("dual_eval_summary.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        loc = obj.get("location") or {}
        sem = obj.get("semantic") or {}
        near = obj.get("near_miss") or {}
        rows.append(
            {
                "label": obj.get("label"),
                "path": str(path.parent),
                "recall": loc.get("recall"),
                "precision": loc.get("precision"),
                "f1": loc.get("f1"),
                "semantic_recall": sem.get("recall"),
                "unmatched_pred_findings": loc.get("location_unmatched_pred_findings"),
                "fn_semantic_near_miss_ratio": near.get("fn_semantic_near_miss_ratio"),
                "fp_rule_too_broad_ratio": near.get("fp_rule_too_broad_ratio"),
                "quote_exact_substring_ratio": (obj.get("quote_quality") or {}).get("quote_exact_substring_ratio"),
            }
        )
    rows.sort(key=lambda x: (-(x.get("precision") or 0), x.get("f1") or 0))
    out = {"variants": rows}
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
