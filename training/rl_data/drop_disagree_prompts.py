#!/usr/bin/env python3
"""Drop DISAGREE sample_ids from RL prompt pools so GRPO does not reward bad gold."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.rl_data.screen_training_data import sample_id

DEFAULT_POOLS = [
    ROOT / "data/rl/rl_prompts.jsonl",
    ROOT / "data/rl/swift_prompts.jsonl",
    ROOT / "data/rl/swift_prompts_max2048.jsonl",
]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def disagree_ids(path: Path) -> Set[str]:
    ids: Set[str] = set()
    for row in _load_jsonl(path):
        sid = sample_id(row)
        if sid:
            ids.add(sid)
    return ids


def filter_pool(rows: List[Dict[str, Any]], drop: Set[str]) -> tuple[List[Dict[str, Any]], int]:
    kept = []
    n_drop = 0
    for row in rows:
        sid = sample_id(row)
        if sid and sid in drop:
            n_drop += 1
            continue
        kept.append(row)
    return kept, n_drop


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--disagree", type=Path, default=ROOT / "data/rl/sft_disagree.jsonl")
    p.add_argument("--pools", type=Path, nargs="*", default=DEFAULT_POOLS)
    p.add_argument("--suffix", default="_no_disagree")
    p.add_argument("--report", type=Path, default=ROOT / "data/rl/disagree_prompt_filter.json")
    args = p.parse_args()
    drop = disagree_ids(args.disagree)
    report: Dict[str, Any] = {"n_disagree_ids": len(drop), "pools": []}
    for pool in args.pools:
        if not pool.is_file():
            continue
        rows = _load_jsonl(pool)
        kept, n_drop = filter_pool(rows, drop)
        out = pool.with_name(pool.stem + args.suffix + pool.suffix)
        with out.open("w", encoding="utf-8") as f:
            for row in kept:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rec = {
            "input": str(pool),
            "output": str(out),
            "n_in": len(rows),
            "n_out": len(kept),
            "n_dropped": n_drop,
        }
        report["pools"].append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
