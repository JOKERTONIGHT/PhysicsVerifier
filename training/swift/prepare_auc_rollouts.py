#!/usr/bin/env python3
"""Build labelled rollouts jsonl for eval_verifier_auc.py.

Sources:
  --predictions  HiPhO/heldout predictions (sample_id, prediction, label)
  --error-eval   error_eval_dataset JSON with physics_error_gt diagnostics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.compat.part_scoring import labels_from_value


def _dump(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def from_predictions(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            sid = str(row.get("sample_id") or meta.get("sample_id") or row.get("id") or "")
            pred = str(row.get("prediction") or row.get("response") or "")
            labels = labels_from_value(row.get("label") if row.get("label") is not None else row.get("answer"))
            out.append(
                {
                    "sample_id": sid,
                    "prediction": pred,
                    "label": labels,
                    "question": row.get("question") or meta.get("question") or "",
                }
            )
    return out


def from_error_eval(path: Path) -> List[Dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    items = blob if isinstance(blob, list) else blob.get("samples") or blob.get("items") or blob.get("data") or []
    out: List[Dict[str, Any]] = []
    for i, row in enumerate(items):
        if not isinstance(row, dict):
            continue
        pred = str(row.get("prediction") or row.get("response") or "")
        if not pred:
            continue
        labels = labels_from_value(row.get("label") if row.get("label") is not None else row.get("answer"))
        diags = []
        for err in row.get("physics_error_gt") or row.get("diagnostics") or []:
            if isinstance(err, dict):
                item = dict(err)
                item.setdefault("severity", "error")
                diags.append(item)
        out.append(
            {
                "sample_id": str(row.get("id") or row.get("sample_id") or f"err{i}"),
                "prediction": pred,
                "label": labels,
                "question": row.get("question") or "",
                "diagnostics": diags,
            }
        )
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--error-eval", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows: List[Dict[str, Any]] = []
    if args.predictions and args.predictions.is_file():
        rows.extend(from_predictions(args.predictions))
    if args.error_eval and args.error_eval.is_file():
        rows.extend(from_error_eval(args.error_eval))
    if not rows:
        print("no rows")
        return 2
    _dump(args.output, rows)
    print(json.dumps({"n": len(rows), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
