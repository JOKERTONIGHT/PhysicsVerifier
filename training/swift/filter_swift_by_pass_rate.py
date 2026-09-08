#!/usr/bin/env python3
"""Keep Swift GRPO prompts whose empirical pass-rate is in (min, max)."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))


def _grade_fallback(resp: str, gold: Any) -> bool:
    from training.compat.math_grading import grade_answer_verl

    return bool(gold) and grade_answer_verl(str(resp), gold)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _key(row: Dict[str, Any]) -> str:
    sid = str(row.get("sample_id") or (row.get("metadata") or {}).get("sample_id") or "")
    if sid:
        return f"id:{sid}"
    return "q:" + str(row.get("question") or "")[:400]


def _label(row: Dict[str, Any]) -> Any:
    return row.get("solution") if row.get("solution") is not None else row.get("label")


def attach_and_filter(
    prompts: List[Dict[str, Any]],
    rollouts: List[Dict[str, Any]],
    *,
    min_pass: float,
    max_pass: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    accs: Dict[str, List[bool]] = defaultdict(list)
    for rec in rollouts:
        key = _key(rec)
        resp = rec.get("response") or rec.get("prediction") or rec.get("completion") or ""
        if rec.get("acc") is not None:
            accs[key].append(bool(rec.get("acc")))
            continue
        gold = rec.get("solution") or rec.get("label")
        accs[key].append(_grade_fallback(str(resp), gold))

    kept: List[Dict[str, Any]] = []
    buckets = {"no_rollout": 0, "too_hard": 0, "too_easy": 0, "kept": 0}
    rates: List[float] = []
    for row in prompts:
        key = _key(row)
        vals = accs.get(key) or []
        if not vals:
            buckets["no_rollout"] += 1
            continue
        pr = sum(1 for a in vals if a) / len(vals)
        rates.append(pr)
        meta = dict(row.get("metadata") or {})
        meta["pass_rate"] = pr
        meta["n_rollouts"] = len(vals)
        out = dict(row)
        out["metadata"] = meta
        if pr <= min_pass:
            buckets["too_hard"] += 1
            continue
        if pr >= max_pass:
            buckets["too_easy"] += 1
            continue
        buckets["kept"] += 1
        kept.append(out)
    audit = {
        "n_prompts": len(prompts),
        "n_rollouts": len(rollouts),
        "n_kept": len(kept),
        "min_pass_rate": min_pass,
        "max_pass_rate": max_pass,
        "mean_pass_rate": sum(rates) / max(len(rates), 1),
        "buckets": buckets,
    }
    return kept, audit


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=ROOT / "data/rl/swift_prompts_max2048.jsonl")
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--output", type=Path, default=ROOT / "data/rl/swift_prompts_hybrid_band.jsonl")
    p.add_argument("--audit", type=Path, default=None)
    p.add_argument("--min-pass-rate", type=float, default=0.05)
    p.add_argument("--max-pass-rate", type=float, default=0.95)
    args = p.parse_args()
    kept, audit = attach_and_filter(
        _load_jsonl(args.prompts),
        _load_jsonl(args.rollouts),
        min_pass=args.min_pass_rate,
        max_pass=args.max_pass_rate,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit_path = args.audit or args.output.with_suffix(".audit.json")
    audit["output"] = str(args.output)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if kept else 2


if __name__ == "__main__":
    raise SystemExit(main())
