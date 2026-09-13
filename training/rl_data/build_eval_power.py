#!/usr/bin/env python3
"""Build an n>=300 same-pool eval set, disjoint from val and olympiad heldout."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sid(row: Dict[str, Any]) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    return str(
        row.get("sample_id")
        or row.get("id")
        or meta.get("sample_id")
        or meta.get("id")
        or ""
    )


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pool", type=Path, default=ROOT / "data/rl/swift_prompts_max2048.jsonl")
    p.add_argument("--exclude", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, default=ROOT / "data/rl/eval_power300.jsonl")
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    if not args.exclude:
        args.exclude = [
            ROOT / "data/rl/heldout_eval_trusted.jsonl",
            ROOT / "data/rl/heldout_eval.jsonl",
            ROOT / "data/rl/val_same_dist.jsonl",
        ]
    blocked: Set[str] = set()
    for path in args.exclude:
        for row in _load_jsonl(path):
            sid = _sid(row)
            if sid:
                blocked.add(sid)
    pool = [r for r in _load_jsonl(args.pool) if _sid(r) not in blocked]
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    chosen = pool[: max(1, args.size)]
    _write_jsonl(args.output, chosen)
    print(
        json.dumps(
            {
                "pool": str(args.pool),
                "n_pool": len(pool),
                "n_blocked": len(blocked),
                "n_eval": len(chosen),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0 if len(chosen) >= min(args.size, 1) else 2


if __name__ == "__main__":
    raise SystemExit(main())
