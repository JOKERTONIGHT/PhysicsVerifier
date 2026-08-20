#!/usr/bin/env python3
"""Subset a full semantic-experience run to the first N expansion samples and rebuild distilled rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_experience_rules import _build_distilled_library, _resume_done_map  # noqa: E402


def _load_expansion_ids(expansion_path: Path, limit: int) -> Set[str]:
    rows = json.loads(expansion_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list: {expansion_path}")
    ids: List[str] = []
    for row in rows[:limit]:
        if isinstance(row, dict) and row.get("id"):
            ids.append(str(row["id"]))
    return set(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Subset semantic experience output by expansion prefix size.")
    parser.add_argument("--semantic-input", type=str, required=True)
    parser.add_argument("--expansion-input", type=str, required=True)
    parser.add_argument("--expansion-size", type=int, required=True)
    parser.add_argument("--semantic-output", type=str, required=True)
    parser.add_argument("--distilled-output", type=str, required=True)
    parser.add_argument("--min-rule-count", type=int, default=1)
    args = parser.parse_args()

    allowed = _load_expansion_ids(Path(args.expansion_input), int(args.expansion_size))
    payload = json.loads(Path(args.semantic_input).read_text(encoding="utf-8"))
    samples = payload.get("samples") if isinstance(payload, dict) else None
    if not isinstance(samples, list):
        raise SystemExit("semantic-input must contain {'samples': [...]}")

    done = _resume_done_map(payload)
    filtered: List[Dict[str, Any]] = []
    for sid, row in done.items():
        if sid in allowed:
            filtered.append(row)

    filtered.sort(key=lambda r: str(r.get("sample_id") or ""))

    out_sem = {"samples": filtered}
    sem_path = Path(args.semantic_output)
    sem_path.parent.mkdir(parents=True, exist_ok=True)
    sem_path.write_text(json.dumps(out_sem, ensure_ascii=False, indent=2), encoding="utf-8")

    distilled = _build_distilled_library(filtered, min_count=max(1, int(args.min_rule_count)))
    dist_path = Path(args.distilled_output)
    dist_path.parent.mkdir(parents=True, exist_ok=True)
    dist_path.write_text(json.dumps(distilled, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "expansion_size": args.expansion_size,
        "allowed_ids": len(allowed),
        "semantic_samples": len(filtered),
        "distilled_rules": distilled.get("summary", {}).get("total_distilled_rules", 0),
        "semantic_output": str(sem_path),
        "distilled_output": str(dist_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
