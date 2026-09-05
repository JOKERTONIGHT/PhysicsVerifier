#!/usr/bin/env python3
"""Quality-check SFT rows: boxed, gold match, no gold-hint leak, not held-out."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.compat.math_grading import grade_answer_verl
from training.rl_data.generate_sft_solutions import HINT_MARK, has_meta_talk, user_turn_has_hint
from training.rl_data.screen_training_data import sft_row_drop_reason

DROP_FLAGS = {
    "empty_assistant",
    "no_boxed",
    "grade_fail",
    "hint_in_user",
    "heldout",
    "meta_talk",
    "visual_input",
    "gold_fit",
    "concatenated_stem",
    "known_unusable",
    "multi_question_single_label",
}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _heldout_ids(path: Optional[Path]) -> Set[str]:
    if path is None or not path.is_file():
        return set()
    ids: Set[str] = set()
    for row in _load_jsonl(path):
        sid = str(row.get("sample_id") or (row.get("metadata") or {}).get("sample_id") or "")
        if sid:
            ids.add(sid)
    return ids


def assistant_text(row: Dict[str, Any]) -> str:
    for msg in reversed(row.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def audit_row(row: Dict[str, Any], heldout_ids: Optional[Set[str]] = None, min_chars: int = 400) -> Dict[str, Any]:
    flags: List[str] = []
    asst = assistant_text(row)
    gold = str(row.get("solution") or "")
    messages = list(row.get("messages") or [])
    if not asst.strip():
        flags.append("empty_assistant")
    if "\\boxed" not in asst:
        flags.append("no_boxed")
    if gold and asst.strip() and not grade_answer_verl(asst, gold):
        flags.append("grade_fail")
    if user_turn_has_hint(messages) or HINT_MARK in str(row.get("question") or ""):
        flags.append("hint_in_user")
    if has_meta_talk(asst):
        flags.append("meta_talk")
    if min_chars and 0 < len(asst) < min_chars:
        flags.append("too_short")
    sid = str(row.get("sample_id") or "")
    if heldout_ids and sid in heldout_ids:
        flags.append("heldout")
    screen = sft_row_drop_reason(row)
    if screen and screen not in flags:
        flags.append(screen)
    drop = any(f in DROP_FLAGS for f in flags)
    return {
        "sample_id": sid,
        "flags": flags,
        "drop": drop,
        "chars": len(asst),
        "generator": row.get("generator"),
        "hint_gold": row.get("hint_gold"),
    }


def audit_rows(
    rows: List[Dict[str, Any]],
    heldout_ids: Optional[Set[str]] = None,
    min_chars: int = 400,
) -> Dict[str, Any]:
    details = [audit_row(r, heldout_ids, min_chars) for r in rows]
    flag_counts: Counter[str] = Counter()
    for d in details:
        flag_counts.update(d["flags"])
    n_drop = sum(1 for d in details if d["drop"])
    n_warn = sum(1 for d in details if d["flags"] and not d["drop"])
    keep_ids = [d["sample_id"] for d in details if not d["drop"]]
    dup = len(keep_ids) - len(set(keep_ids))
    return {
        "n": len(rows),
        "n_ok": sum(1 for d in details if not d["flags"]),
        "n_warn": n_warn,
        "n_drop": n_drop,
        "n_duplicate_ids": max(dup, 0),
        "flag_counts": dict(flag_counts),
        "rows": details,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "data/rl/sft_solutions.jsonl")
    p.add_argument("--heldout", type=Path, default=ROOT / "data/rl/heldout_eval.jsonl")
    p.add_argument("--report", type=Path, default=ROOT / "data/rl/sft_quality_report.json")
    p.add_argument("--rejected", type=Path, default=ROOT / "data/rl/sft_solutions_rejected.jsonl")
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--apply", action="store_true", help="Rewrite --input keeping non-drop rows")
    args = p.parse_args()

    rows = _load_jsonl(args.input)
    report = audit_rows(rows, _heldout_ids(args.heldout), args.min_chars)
    details = report.pop("rows")
    public = dict(report)
    public["input"] = str(args.input)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({**public, "rows": details}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(public, ensure_ascii=False, indent=2))

    if args.apply and report["n_drop"]:
        keep: List[Dict[str, Any]] = []
        drop: List[Dict[str, Any]] = []
        for row, detail in zip(rows, details):
            if detail["drop"]:
                rec = dict(row)
                rec["audit_flags"] = detail["flags"]
                drop.append(rec)
            else:
                keep.append(row)
        args.rejected.parent.mkdir(parents=True, exist_ok=True)
        with args.rejected.open("a", encoding="utf-8") as f:
            for rec in drop:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp = args.input.with_suffix(args.input.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for rec in keep:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        tmp.replace(args.input)
        print(json.dumps({"rewrote": str(args.input), "kept": len(keep), "rejected": len(drop)}, ensure_ascii=False))
    return 0 if report["n_drop"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
