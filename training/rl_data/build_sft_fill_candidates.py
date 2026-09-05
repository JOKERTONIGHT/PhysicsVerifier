#!/usr/bin/env python3
"""Rank unlabeled, text-only, gradeable stems for SFT fill (target ~300 rows)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.compat.math_grading import extract_answer
from training.rl_data.screen_training_data import MULTI_ASK_RE, looks_multi_question, prompt_drop_reason

EXTRA_VIS = re.compile(
    r"(?i)(map shown|diagram depicts|the following diagram|"
    r"attached is a plot|photograph|table below|equipment:|"
    r"see the map)"
)
FRAG = re.compile(
    r"(?i)(in question\s+\d|previous part|as in part|the current in question|"
    r"from the previous|using the result)"
)


def _ids(path: Path) -> set[str]:
    out: set[str] = set()
    if not path.is_file():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        out.add(str(json.loads(line).get("sample_id") or ""))
    return out


def gradeable(gold: str) -> bool:
    g = str(gold or "")
    if not g.strip():
        return False
    if "\\boxed" in g:
        return extract_answer(g) is not None
    return True


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=ROOT / "data/rl/swift_prompts.jsonl")
    p.add_argument("--sft", type=Path, default=ROOT / "data/rl/sft_solutions.jsonl")
    p.add_argument("--heldout", type=Path, default=ROOT / "data/rl/heldout_eval.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/rl/sft_fill_candidates.jsonl")
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--min-chars", type=int, default=400)
    p.add_argument("--max-chars", type=int, default=1400)
    args = p.parse_args()
    if not args.src.is_file():
        print(json.dumps({"error": f"missing {args.src}", "n_written": 0}))
        return 2
    done = _ids(args.sft)
    held = _ids(args.heldout)
    ranked = []
    for line in args.src.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        sid = str(row.get("sample_id") or "")
        if not sid or sid in done or sid in held:
            continue
        if prompt_drop_reason(row):
            continue
        q = str(row.get("question") or "")
        gold = str(row.get("solution") or "")
        if not (args.min_chars <= len(q) <= args.max_chars):
            continue
        if not gradeable(gold):
            continue
        if EXTRA_VIS.search(q) or FRAG.search(q):
            continue
        if looks_multi_question(q, 1):
            continue
        n_ask = len(MULTI_ASK_RE.findall(q))
        n_q = q.count("?")
        score = abs(len(q) - 520) + 80 * max(n_ask - 1, 0) + 40 * max(n_q - 1, 0)
        ranked.append((score, row))
    ranked.sort(key=lambda x: x[0])
    out = [t[1] for t in ranked[: args.limit]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"n_ranked": len(ranked), "n_written": len(out), "n_done_sft": len(done), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
