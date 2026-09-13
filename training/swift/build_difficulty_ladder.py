#!/usr/bin/env python3
"""Write a difficulty ladder table from per-tier score JSON files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--easy", type=Path, default=ROOT / "results/tier_ladder/easy/heldout_scores.json")
    p.add_argument("--mid", type=Path, default=ROOT / "results/tier_ladder/mid/heldout_scores.json")
    p.add_argument("--hard", type=Path, default=ROOT / "results/tier_ladder/hard/heldout_scores.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/tier_ladder/ladder.json")
    args = p.parse_args()
    rows: List[Dict[str, Any]] = []
    for tier, path in (("easy", args.easy), ("mid", args.mid), ("hard", args.hard)):
        scores = _load(path)
        if not scores:
            rows.append({"tier": tier, "missing": True, "path": str(path)})
            continue
        part = float(scores.get("part_avg_at_k") or 0.0)
        verdict = "ok"
        if tier == "easy":
            if part > 0.85:
                verdict = "too_easy_smoke_only"
            elif part < 0.4:
                verdict = "too_hard_for_easy_band"
            else:
                verdict = "in_band_0.4_0.7"
        rows.append(
            {
                "tier": tier,
                "n_samples": scores.get("n_samples"),
                "k": scores.get("k"),
                "part_avg_at_k": part,
                "part_avg_at_k_ci": scores.get("part_avg_at_k_ci"),
                "part_pass_at_k": scores.get("part_pass_at_k"),
                "part_pass_minus_avg": scores.get("part_pass_minus_avg"),
                "item_pass_at_k": scores.get("item_pass_at_k"),
                "item_avg_at_k": scores.get("item_avg_at_k"),
                "verdict": verdict,
                "path": str(path),
            }
        )
    report = {"rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
