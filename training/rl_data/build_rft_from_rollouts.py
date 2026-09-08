#!/usr/bin/env python3
"""Build on-policy RFT jsonl from correct base-model rollouts."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.rl_data.screen_training_data import sft_style_drop_reason

SFT_SYSTEM = (
    "You are an expert physics competition solver. "
    "Show clear step-by-step reasoning and put the final answer in \\boxed{}."
)
TARGET_LEN = 4600


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sid(row: Dict[str, Any]) -> str:
    return str(row.get("sample_id") or (row.get("metadata") or {}).get("sample_id") or "")


def _prompt_index(prompts: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in prompts:
        sid = _sid(row)
        if sid:
            out[sid] = row
    return out


def _messages_for(prompt: Optional[Dict[str, Any]], question: str, solution: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if prompt:
        for msg in prompt.get("messages") or prompt.get("input") or []:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "user")
            if role == "assistant":
                continue
            msgs.append({"role": role, "content": str(msg.get("content") or "")})
    if not msgs:
        msgs = [
            {"role": "system", "content": SFT_SYSTEM},
            {"role": "user", "content": question},
        ]
    msgs.append({"role": "assistant", "content": solution})
    return msgs


def select_rollouts(
    rollouts: List[Dict[str, Any]],
    *,
    max_per_id: int = 2,
    target_len: int = TARGET_LEN,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    n_correct = 0
    n_style_drop = 0
    for rec in rollouts:
        if not rec.get("acc") and float(rec.get("part_frac") or 0) < 1.0:
            continue
        n_correct += 1
        text = str(rec.get("response") or rec.get("prediction") or rec.get("completion") or "")
        reason = sft_style_drop_reason(text, mode="rft", min_chars=400, max_chars=8000)
        if reason:
            n_style_drop += 1
            continue
        sid = _sid(rec)
        if not sid:
            continue
        by_id[sid].append(rec)

    kept: List[Dict[str, Any]] = []
    for sid, recs in by_id.items():
        recs = sorted(recs, key=lambda r: abs(len(str(r.get("response") or r.get("prediction") or "")) - target_len))
        kept.extend(recs[: max(1, max_per_id)])
    audit = {
        "n_rollouts": len(rollouts),
        "n_correct": n_correct,
        "n_style_drop": n_style_drop,
        "n_ids_with_correct": len(by_id),
        "n_kept": len(kept),
        "max_per_id": max_per_id,
        "target_len": target_len,
    }
    return kept, audit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--prompts", type=Path, default=ROOT / "data/rl/sft_solutions.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/rl/rft_solutions.jsonl")
    p.add_argument("--audit", type=Path, default=ROOT / "data/rl/rft_build_report.json")
    p.add_argument("--max-per-id", type=int, default=2)
    p.add_argument("--target-len", type=int, default=TARGET_LEN)
    args = p.parse_args()

    prompts = _prompt_index(_load_jsonl(args.prompts))
    kept, audit = select_rollouts(
        _load_jsonl(args.rollouts),
        max_per_id=args.max_per_id,
        target_len=args.target_len,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.output.open("w", encoding="utf-8") as fout:
        for rec in kept:
            sid = _sid(rec)
            prompt = prompts.get(sid)
            question = str(rec.get("question") or (prompt or {}).get("question") or "")
            text = str(rec.get("response") or rec.get("prediction") or "")
            gold = rec.get("solution") or (prompt or {}).get("solution") or ""
            row = {
                "messages": _messages_for(prompt, question, text),
                "solution": gold,
                "question": question,
                "sample_id": sid,
                "source": (prompt or {}).get("source") or "rft_rollout",
                "generator": "base_rft",
                "rollout_index": rec.get("rollout_index"),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
    audit["n_written"] = n_written
    audit["output"] = str(args.output)
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0 if n_written else 2


if __name__ == "__main__":
    raise SystemExit(main())
