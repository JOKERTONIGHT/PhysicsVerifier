#!/usr/bin/env python3
"""Seed long-chain SFT rows from base correct rollouts (style anchors).

API rewrite can then fill remaining ids. Existing output rows are kept.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.rl_data.build_rft_from_rollouts import _messages_for, _sid, select_rollouts
from training.rl_data.screen_training_data import sample_id


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _done_ids(path: Path) -> Set[str]:
    return {sample_id(r) for r in _load_jsonl(path) if sample_id(r)}


def seed(
    prompts: List[Dict[str, Any]],
    rollouts: List[Dict[str, Any]],
    *,
    done: Set[str],
    target_len: int = 4500,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    want = {sample_id(r) for r in prompts if sample_id(r)}
    by_prompt = {sample_id(r): r for r in prompts if sample_id(r)}
    kept, audit = select_rollouts(rollouts, max_per_id=1, target_len=target_len)
    rows: List[Dict[str, Any]] = []
    for rec in kept:
        sid = _sid(rec)
        if not sid or sid not in want or sid in done:
            continue
        prompt = by_prompt.get(sid) or {}
        text = str(rec.get("response") or rec.get("prediction") or "")
        question = str(rec.get("question") or prompt.get("question") or "")
        rows.append(
            {
                "messages": _messages_for(prompt, question, text),
                "solution": prompt.get("solution") or rec.get("solution") or "",
                "question": question,
                "sample_id": sid,
                "source": prompt.get("source") or "base_rollout_anchor",
                "generator": "base_rollout_anchor",
                "hint_gold": False,
            }
        )
    audit["n_prompt_ids"] = len(want)
    audit["n_already_done"] = len(done)
    audit["n_seeded"] = len(rows)
    return rows, audit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=ROOT / "data/rl/sft_longchain_prompts.jsonl")
    p.add_argument("--rollouts", type=Path, default=ROOT / "data/rl/base_pass_rates_sft554.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/rl/sft_solutions_longchain.jsonl")
    p.add_argument("--report", type=Path, default=ROOT / "data/rl/sft_longchain_anchor_report.json")
    p.add_argument("--target-len", type=int, default=4500)
    args = p.parse_args()
    prompts = _load_jsonl(args.prompts)
    done = _done_ids(args.output)
    rows, audit = seed(
        prompts,
        _load_jsonl(args.rollouts),
        done=done,
        target_len=args.target_len,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit["output"] = str(args.output)
    args.report.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if (rows or done) else 2


if __name__ == "__main__":
    raise SystemExit(main())
