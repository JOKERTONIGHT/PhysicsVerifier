#!/usr/bin/env python3
"""Extract a small, deduplicated wrong-answer set for incremental-rule testing."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.combined_language_io import iter_rollout_batches  # noqa: E402
from scripts.combined_language_samples import (  # noqa: E402
    sample_to_eval_row,
    stable_question_key,
)


def _load_excluded_question_keys(paths: Iterable[Path]) -> Set[str]:
    keys: Set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Exclude file must contain a JSON list: {path}")
        for row in payload:
            if isinstance(row, dict) and str(row.get("question") or "").strip():
                keys.add(stable_question_key(str(row["question"])))
    return keys


def _is_complete_wrong_row(row: Dict[str, Any]) -> bool:
    return (
        row.get("source_reward_acc") is False
        and len(str(row.get("question") or "").strip()) >= 40
        and len(str(row.get("prediction") or "").strip()) >= 80
        and bool(str(row.get("answer") or "").strip())
        and str(row.get("answer") or "").strip() != "[]"
    )


def extract_incremental_samples(
    *,
    input_path: Path,
    exclude_paths: Iterable[Path],
    target_size: int,
    max_rollouts: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if target_size <= 0:
        raise ValueError("target_size must be positive")

    rng = random.Random(seed)
    excluded_keys = _load_excluded_question_keys(exclude_paths)
    seen_question_keys = set(excluded_keys)
    reservoir: List[Dict[str, Any]] = []
    eligible_questions = 0
    rollouts_seen = 0
    raw_samples_seen = 0

    for batch in iter_rollout_batches(input_path):
        rollout_id = batch.get("rollout_id")
        per_question: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        for seq_index, sample in enumerate(batch.get("samples") or []):
            raw_samples_seen += 1
            if not isinstance(sample, dict):
                continue
            row = sample_to_eval_row(rollout_id, sample, seq_index=seq_index)
            if not _is_complete_wrong_row(row):
                continue
            question_key = str((row.get("meta") or {}).get("question_key") or "")
            if not question_key or question_key in seen_question_keys:
                continue

            count, _ = per_question.get(question_key, (0, row))
            count += 1
            if count == 1 or rng.randrange(count) == 0:
                per_question[question_key] = (count, row)
            else:
                per_question[question_key] = (count, per_question[question_key][1])

        for question_key, (_, row) in sorted(per_question.items()):
            seen_question_keys.add(question_key)
            eligible_questions += 1
            if len(reservoir) < target_size:
                reservoir.append(row)
            else:
                replacement = rng.randrange(eligible_questions)
                if replacement < target_size:
                    reservoir[replacement] = row

        rollouts_seen += 1
        if max_rollouts > 0 and rollouts_seen >= max_rollouts:
            break

    rows = sorted(reservoir, key=lambda item: str(item.get("id") or ""))
    report = {
        "input": str(input_path),
        "target_size": target_size,
        "actual_size": len(rows),
        "seed": seed,
        "max_rollouts": max_rollouts or None,
        "rollouts_seen": rollouts_seen,
        "raw_samples_seen": raw_samples_seen,
        "excluded_question_count": len(excluded_keys),
        "eligible_unique_questions": eligible_questions,
        "all_selected_are_wrong": all(
            row.get("source_reward_acc") is False for row in rows
        ),
        "selected_ids": [row.get("id") for row in rows],
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract unique new wrong-answer samples for an incremental-rule smoke test."
    )
    parser.add_argument(
        "--input",
        default="data/combined_language_only.json",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Existing evaluation JSON list to exclude by normalized question; repeatable.",
    )
    parser.add_argument(
        "--output",
        default="data/incremental_rule_test_samples.json",
    )
    parser.add_argument(
        "--report",
        default="results/unified_rules_incremental/sample_selection_report.json",
    )
    parser.add_argument("--target-size", type=int, default=8)
    parser.add_argument("--max-rollouts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    exclude_paths = [Path(value) for value in args.exclude]
    if not exclude_paths:
        exclude_paths = [Path("data/evaluation_sample_3000_expansion.json")]

    rows, report = extract_incremental_samples(
        input_path=Path(args.input),
        exclude_paths=exclude_paths,
        target_size=args.target_size,
        max_rollouts=args.max_rollouts,
        seed=args.seed,
    )
    if len(rows) < args.target_size:
        raise SystemExit(
            f"Only found {len(rows)} eligible unique questions; increase --max-rollouts."
        )

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
