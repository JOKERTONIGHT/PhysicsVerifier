#!/usr/bin/env python3
"""Prepare leak-free rule-expansion pool and eval holdout for scale-curve experiments.

Policy:
- Eval holdout: dual-chain ``main_test`` (200 samples), disjoint from rule expansion.
- Rule expansion: wrong-answer samples from ``combined_language_only.json``, excluding
  all held-out split IDs (main_test + val_ablation + smoke).
- Checkpoint slices: prefix subsets of the ordered expansion pool for scale curves.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.combined_language_io import iter_rollout_batches  # noqa: E402
from scripts.combined_language_samples import sample_to_eval_row  # noqa: E402


def _load_ids(path: Path) -> Set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return {str(row.get("id") or "").strip() for row in data if isinstance(row, dict) and row.get("id")}


def _is_wrong(row: Dict[str, Any]) -> bool:
    acc = row.get("source_reward_acc")
    if acc is True:
        return False
    if acc is False:
        return True
    score = row.get("source_reward_score")
    if isinstance(score, (int, float)) and score < 0.999999:
        return True
    return True


def _collect_holdout_ids(dual_chain_dir: Path) -> Set[str]:
    qa = dual_chain_dir / "qa_chain"
    holdout: Set[str] = set()
    for name in ("combined_language_main_test.json", "combined_language_val_ablation.json", "combined_language_smoke.json"):
        path = qa / name
        if path.exists():
            holdout |= _load_ids(path)
    return holdout


def _load_rule_expansion_base(dual_chain_dir: Path) -> List[Dict[str, Any]]:
    path = dual_chain_dir / "qa_chain" / "combined_language_rule_expansion.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list: {path}")
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = {k: row[k] for k in ("id", "question", "prediction", "answer") if k in row}
        if item.get("id") and item.get("question") and item.get("prediction"):
            cleaned.append(item)
    return cleaned


def _stream_extra_wrong(
    combined_path: Path,
    *,
    holdout_ids: Set[str],
    existing_ids: Set[str],
    target_total: int,
) -> List[Dict[str, Any]]:
    extra: List[Dict[str, Any]] = []
    seen = set(existing_ids)
    for batch in iter_rollout_batches(combined_path):
        rid = batch.get("rollout_id")
        for i, sample in enumerate(batch.get("samples") or []):
            if not isinstance(sample, dict):
                continue
            row = sample_to_eval_row(rid, sample, seq_index=i)
            sid = str(row.get("id") or "").strip()
            if not sid or sid in seen or sid in holdout_ids:
                continue
            if not _is_wrong(row):
                continue
            if not str(row.get("question") or "").strip() or not str(row.get("prediction") or "").strip():
                continue
            item = {
                "id": sid,
                "question": row["question"],
                "prediction": row["prediction"],
                "answer": row.get("answer", ""),
            }
            extra.append(item)
            seen.add(sid)
            if len(existing_ids) + len(extra) >= target_total:
                return extra
    return extra


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leak-free expansion pool and eval holdout.")
    parser.add_argument(
        "--dual-chain-dir",
        type=str,
        default=str(REPO_ROOT / "data/derived/combined_language_dual_chain_seed20260508_test200"),
    )
    parser.add_argument(
        "--combined-input",
        type=str,
        default=str(REPO_ROOT / "data/combined_language_only.json"),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(REPO_ROOT / "data/derived/leak_free_scale_seed20260508"),
    )
    parser.add_argument("--expansion-target", type=int, default=3000)
    parser.add_argument(
        "--checkpoint-sizes",
        type=str,
        default="300,600,900,1200,1500,1800,2100,2400,2700,3000",
        help="Comma-separated expansion sample counts for checkpoint slices.",
    )
    parser.add_argument(
        "--legacy-error-eval",
        type=str,
        default=str(
            REPO_ROOT
            / "data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain/error_eval_dataset_100.json"
        ),
    )
    args = parser.parse_args()

    dual_chain_dir = Path(args.dual_chain_dir)
    outdir = Path(args.outdir)
    checkpoint_sizes = sorted({int(x.strip()) for x in args.checkpoint_sizes.split(",") if x.strip()})

    holdout_ids = _collect_holdout_ids(dual_chain_dir)
    expansion_base = _load_rule_expansion_base(dual_chain_dir)
    base_ids = {str(r["id"]) for r in expansion_base}

    overlap_base_holdout = sorted(base_ids & holdout_ids)
    if overlap_base_holdout:
        raise SystemExit(f"rule_expansion overlaps holdout: {overlap_base_holdout[:5]}")

    target = int(args.expansion_target)
    pool = list(expansion_base)
    if len(pool) < target:
        extra = _stream_extra_wrong(
            Path(args.combined_input),
            holdout_ids=holdout_ids,
            existing_ids=base_ids,
            target_total=target,
        )
        pool.extend(extra)

    pool = pool[:target]
    pool_ids = [str(r["id"]) for r in pool]

    eval_holdout_path = dual_chain_dir / "qa_chain" / "combined_language_main_test.json"
    eval_holdout = json.loads(eval_holdout_path.read_text(encoding="utf-8"))
    eval_ids = _load_ids(eval_holdout_path)

    error_eval_path = Path(args.legacy_error_eval)
    error_eval_ids: Set[str] = set()
    error_eval_overlap_expansion: Set[str] = set()
    error_eval_outside_holdout: Set[str] = set()
    if error_eval_path.exists():
        error_eval_ids = _load_ids(error_eval_path)
        error_eval_overlap_expansion = error_eval_ids & set(pool_ids)
        error_eval_outside_holdout = error_eval_ids - eval_ids

    manifest = {
        "policy": {
            "eval_holdout_200": "main_test (200 samples) reserved for future eval; never used for rule mining",
            "error_eval_100": "current error-level eval set (100 samples), subset of eval_holdout_200",
            "rule_expansion": "rule_expansion split + additional wrong answers excluding holdout",
            "holdout_splits": ["main_test", "val_ablation", "smoke"],
        },
        "counts": {
            "holdout_ids": len(holdout_ids),
            "expansion_pool": len(pool),
            "eval_holdout_200": len(eval_holdout),
            "error_eval_100": len(error_eval_ids),
            "expansion_from_dual_chain_base": len(expansion_base),
            "expansion_extra_streamed": max(0, len(pool) - len(expansion_base)),
        },
        "overlap_audit": {
            "expansion_vs_holdout": sorted(set(pool_ids) & holdout_ids),
            "expansion_vs_eval_holdout_200": sorted(set(pool_ids) & eval_ids),
            "error_eval_100_vs_expansion": sorted(error_eval_overlap_expansion),
            "error_eval_100_outside_holdout_200": sorted(error_eval_outside_holdout),
            "error_eval_100_subset_of_holdout_200": error_eval_ids <= eval_ids if error_eval_ids else None,
            "passes": not (set(pool_ids) & holdout_ids)
            and not error_eval_overlap_expansion
            and (not error_eval_ids or error_eval_ids <= eval_ids),
        },
        "paths": {
            "expansion_pool": str(outdir / "expansion_pool.json"),
            "eval_holdout_200": str(outdir / "eval_holdout_200.json"),
            "error_eval_100": str(error_eval_path),
            "checkpoints_dir": str(outdir / "checkpoints"),
        },
        "checkpoint_sizes": checkpoint_sizes,
    }

    _write_json(outdir / "expansion_pool.json", pool)
    _write_json(outdir / "eval_holdout_200.json", eval_holdout)
    _write_json(outdir / "split_manifest.json", manifest)

    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for n in checkpoint_sizes:
        if n > len(pool):
            continue
        _write_json(ckpt_dir / f"expansion_sample_{n:04d}.json", pool[:n])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
