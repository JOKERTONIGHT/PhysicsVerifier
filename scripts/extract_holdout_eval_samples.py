#!/usr/bin/env python3
"""Extract a deduplicated correct/wrong holdout pool from combined-language rollouts."""

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
from scripts.combined_language_samples import sample_to_eval_row, stable_question_key  # noqa: E402


DEFAULT_EXCLUDES = (
    "data/evaluation_sample_300.json",
    "data/evaluation_sample_3000_expansion.json",
    "data/typical_samples.json",
    "data/incremental_rule_test_samples.json",
)


def resolve_exclude_paths(explicit_paths: Iterable[Path], *, include_defaults: bool = True) -> List[Path]:
    """Combine the permanent leakage exclusions with run-specific used datasets."""
    candidates: List[Path] = []
    if include_defaults:
        candidates.extend(Path(value) for value in DEFAULT_EXCLUDES)
    candidates.extend(Path(value) for value in explicit_paths)
    out: List[Path] = []
    seen: Set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def load_excluded_question_keys(paths: Iterable[Path]) -> Set[str]:
    keys: Set[str] = set()
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"Exclude file must contain a JSON list: {path}")
        for row in payload:
            if not isinstance(row, dict):
                continue
            question = str(row.get("question") or "").strip()
            if question:
                keys.add(stable_question_key(question))
    return keys


def _complete_row(row: Dict[str, Any]) -> bool:
    return (
        len(str(row.get("question") or "").strip()) >= 40
        and len(str(row.get("prediction") or "").strip()) >= 80
        and bool(str(row.get("answer") or "").strip())
        and str(row.get("answer") or "").strip() not in {"[]", "null"}
        and isinstance(row.get("source_reward_acc"), bool)
    )


def _reservoir_add(
    pool: List[Dict[str, Any]],
    row: Dict[str, Any],
    *,
    eligible_count: int,
    target_size: int,
    rng: random.Random,
) -> None:
    if len(pool) < target_size:
        pool.append(row)
        return
    replacement = rng.randrange(eligible_count)
    if replacement < target_size:
        pool[replacement] = row


def extract_holdout_samples(
    *,
    input_path: Path,
    exclude_paths: Iterable[Path],
    wrong_size: int,
    correct_size: int,
    max_rollouts: int,
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if wrong_size < 0 or correct_size < 0 or wrong_size + correct_size <= 0:
        raise ValueError("wrong_size and correct_size must be non-negative with a positive total")

    excluded_keys = load_excluded_question_keys(exclude_paths)
    seen_keys = set(excluded_keys)
    wrong_pool: List[Dict[str, Any]] = []
    correct_pool: List[Dict[str, Any]] = []
    wrong_eligible = 0
    correct_eligible = 0
    raw_samples_seen = 0
    rollouts_seen = 0
    rng = random.Random(seed)

    for batch in iter_rollout_batches(input_path):
        rollout_id = batch.get("rollout_id")
        rows_by_question: Dict[str, List[Dict[str, Any]]] = {}
        for seq_index, sample in enumerate(batch.get("samples") or []):
            raw_samples_seen += 1
            if not isinstance(sample, dict):
                continue
            row = sample_to_eval_row(rollout_id, sample, seq_index=seq_index)
            if not _complete_row(row):
                continue
            question_key = str((row.get("meta") or {}).get("question_key") or "")
            if not question_key or question_key in seen_keys:
                continue
            rows_by_question.setdefault(question_key, []).append(row)

        for question_key, candidates in sorted(rows_by_question.items()):
            if question_key in seen_keys:
                continue
            wrong_candidates = [row for row in candidates if row.get("source_reward_acc") is False]
            correct_candidates = [row for row in candidates if row.get("source_reward_acc") is True]

            # Prefer the class that still has fewer eligible unique questions, which
            # keeps the final pool balanced when a question has both correct and wrong rollouts.
            if wrong_candidates and correct_candidates:
                choose_wrong = (wrong_eligible / max(1, wrong_size)) <= (
                    correct_eligible / max(1, correct_size)
                )
            else:
                choose_wrong = bool(wrong_candidates)

            if choose_wrong and wrong_candidates:
                row = rng.choice(wrong_candidates)
                wrong_eligible += 1
                _reservoir_add(
                    wrong_pool,
                    row,
                    eligible_count=wrong_eligible,
                    target_size=wrong_size,
                    rng=rng,
                )
            elif correct_candidates:
                row = rng.choice(correct_candidates)
                correct_eligible += 1
                _reservoir_add(
                    correct_pool,
                    row,
                    eligible_count=correct_eligible,
                    target_size=correct_size,
                    rng=rng,
                )
            seen_keys.add(question_key)

        rollouts_seen += 1
        if max_rollouts > 0 and rollouts_seen >= max_rollouts:
            break

    wrong_pool = sorted(wrong_pool, key=lambda row: str(row.get("id") or ""))
    correct_pool = sorted(correct_pool, key=lambda row: str(row.get("id") or ""))
    rows = wrong_pool + correct_pool
    report = {
        "input": str(input_path),
        "seed": seed,
        "max_rollouts": max_rollouts or None,
        "rollouts_seen": rollouts_seen,
        "raw_samples_seen": raw_samples_seen,
        "excluded_question_count": len(excluded_keys),
        "wrong_target_size": wrong_size,
        "wrong_actual_size": len(wrong_pool),
        "wrong_eligible_unique_questions": wrong_eligible,
        "correct_target_size": correct_size,
        "correct_actual_size": len(correct_pool),
        "correct_eligible_unique_questions": correct_eligible,
        "question_keys_unique": len(
            {
                str((row.get("meta") or {}).get("question_key") or "")
                for row in rows
                if str((row.get("meta") or {}).get("question_key") or "")
            }
        )
        == len(rows),
        "selected_wrong_ids": [row.get("id") for row in wrong_pool],
        "selected_correct_ids": [row.get("id") for row in correct_pool],
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a question-deduplicated held-out pool of correct and wrong answers."
    )
    parser.add_argument("--input", default="data/combined_language_only.json")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Additional used dataset to exclude by question; repeatable and additive to defaults.",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Disable the permanent default exclusions (unsafe for final evaluation unless replaced explicitly).",
    )
    parser.add_argument("--output", default="data/holdout_eval_candidates.json")
    parser.add_argument("--report", default="results/holdout_eval_candidates_report.json")
    parser.add_argument("--wrong-size", type=int, default=20)
    parser.add_argument("--correct-size", type=int, default=10)
    parser.add_argument("--max-rollouts", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args()

    exclude_paths = resolve_exclude_paths(
        [Path(value) for value in args.exclude],
        include_defaults=not args.no_default_excludes,
    )

    rows, report = extract_holdout_samples(
        input_path=Path(args.input),
        exclude_paths=exclude_paths,
        wrong_size=args.wrong_size,
        correct_size=args.correct_size,
        max_rollouts=args.max_rollouts,
        seed=args.seed,
    )
    if report["wrong_actual_size"] < args.wrong_size or report["correct_actual_size"] < args.correct_size:
        raise SystemExit(
            "Insufficient held-out samples: "
            f"wrong={report['wrong_actual_size']}/{args.wrong_size}, "
            f"correct={report['correct_actual_size']}/{args.correct_size}. "
            "Increase --max-rollouts."
        )

    output_path = Path(args.output)
    report_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
