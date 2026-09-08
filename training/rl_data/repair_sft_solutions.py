#!/usr/bin/env python3
"""Strip gold-hint meta-talk from SFT assistants and keep only rows that still grade.

Used after hint-gold API fills, where models often write 'the reference answer is...'.
Does not invent physics: it only deletes leaking sentences, then re-grades.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.compat.math_grading import grade_answer_verl
from training.rl_data.audit_sft_solutions import assistant_text, audit_rows
from training.rl_data.generate_sft_solutions import META_TALK_RE, has_meta_talk

PARA_SPLIT = re.compile(r"\n\s*\n")
# Drop a whole paragraph if it talks about a provided/reference answer.
LEAK_PARA = re.compile(
    r"reference answer|given answer|the given reference|following the given|"
    r"as provided|internal target|the target (answer|result)",
    re.IGNORECASE,
)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def set_assistant(row: Dict[str, Any], text: str) -> Dict[str, Any]:
    out = dict(row)
    messages = [dict(m) for m in (row.get("messages") or [])]
    replaced = False
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            messages[i]["content"] = text
            replaced = True
            break
    if not replaced:
        messages.append({"role": "assistant", "content": text})
    out["messages"] = messages
    return out


def strip_meta_talk(text: str) -> str:
    paras = PARA_SPLIT.split(text or "")
    kept = [p.strip() for p in paras if p.strip() and not LEAK_PARA.search(p)]
    # Also drop leftover leaky sentences inside surviving paragraphs.
    cleaned: List[str] = []
    for p in kept:
        sents = re.split(r"(?<=[.!?])\s+", p)
        sents = [s for s in sents if s and not LEAK_PARA.search(s)]
        if sents:
            cleaned.append(" ".join(sents))
    return re.sub(r"\n{3,}", "\n\n", "\n\n".join(cleaned)).strip()


def repair_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    asst = assistant_text(row)
    gold = str(row.get("solution") or "")
    if not has_meta_talk(asst) and not META_TALK_RE.search(asst):
        return row, "ok"
    stripped = strip_meta_talk(asst)
    if not stripped or "\\boxed" not in stripped:
        return row, "unrepairable"
    if gold and not grade_answer_verl(stripped, gold):
        return row, "unrepairable"
    if has_meta_talk(stripped):
        return row, "unrepairable"
    rec = set_assistant(row, stripped)
    rec["repaired"] = "strip_meta_talk"
    return rec, "repaired"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "data/rl/sft_solutions.jsonl")
    p.add_argument("--output", type=Path, default=None, help="Default: overwrite --input")
    p.add_argument("--rejected", type=Path, default=ROOT / "data/rl/sft_solutions_rejected.jsonl")
    p.add_argument("--report", type=Path, default=ROOT / "data/rl/sft_repair_report.json")
    p.add_argument("--heldout", type=Path, default=ROOT / "data/rl/heldout_eval.jsonl")
    args = p.parse_args()
    out_path = args.output or args.input

    rows = _load_jsonl(args.input)
    keep: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    counts = {"ok": 0, "repaired": 0, "unrepairable": 0}
    for row in rows:
        rec, status = repair_row(row)
        counts[status] += 1
        if status == "unrepairable":
            bad = dict(row)
            bad["audit_flags"] = ["meta_talk_unrepairable"]
            rejected.append(bad)
        else:
            keep.append(rec)

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for rec in keep:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(out_path)
    if rejected:
        args.rejected.parent.mkdir(parents=True, exist_ok=True)
        with args.rejected.open("a", encoding="utf-8") as f:
            for rec in rejected:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    from training.rl_data.audit_sft_solutions import _heldout_ids

    audit = audit_rows(keep, _heldout_ids(args.heldout), min_chars=400)
    details = audit.pop("rows")
    report = {
        "input": str(args.input),
        "output": str(out_path),
        "n_in": len(rows),
        "n_out": len(keep),
        "repair_counts": counts,
        "post_audit": audit,
    }
    args.report.write_text(
        json.dumps({**report, "rows": details}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if audit.get("n_drop", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
