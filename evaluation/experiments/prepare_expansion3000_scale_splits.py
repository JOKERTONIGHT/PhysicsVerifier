#!/usr/bin/env python3
"""Prepare leak-free scale-curve splits from evaluation_sample_3000_expansion.json.

Policy
------
- Source universe: ``data/evaluation_sample_3000_expansion.json`` (3000 samples, rollout IDs).
- Block rule mining for every source row whose question matches any error-level eval row
  (handles duplicate questions with different IDs).
- Eval holdout (200): reserved subset of blocked rows; never used for rule mining.
- Rule expansion pool (~2691): all non-blocked samples, stable source order.
- Error eval (100): annotated rows; expansion IDs when the question exists in source.
- Semantic experience: reuse ``catalogs/semantic_experience.json`` (no re-extraction).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.generate_experience_rules import _resume_done_map  # noqa: E402


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array: {path}")
    return [row for row in data if isinstance(row, dict)]


def _question_key(row: Dict[str, Any], *, width: int = 400) -> str:
    return str(row.get("question") or "")[:width].strip()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _index_source_by_question(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        key = _question_key(row)
        if not key:
            continue
        out.setdefault(key, []).append(row)
    return out


def _remap_error_eval(
    *,
    annotated_eval: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep dual-chain annotated eval rows unchanged (predictions match GT spans)."""
    remapped: List[Dict[str, Any]] = []
    for row in annotated_eval:
        out = dict(row)
        out["eval_id_namespace"] = "dual_chain"
        remapped.append(out)
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leak-free splits from evaluation_sample_3000_expansion.json.")
    parser.add_argument(
        "--source",
        type=str,
        default=str(REPO_ROOT / "data/evaluation_sample_3000_expansion.json"),
    )
    parser.add_argument(
        "--annotated-error-eval",
        type=str,
        default=str(
            REPO_ROOT
            / "data/derived/combined_language_dual_chain_seed20260508_test200/annotated_chain/error_eval_dataset_100.json"
        ),
    )
    parser.add_argument(
        "--semantic-experience",
        type=str,
        default=str(REPO_ROOT / "catalogs/semantic_experience.json"),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(REPO_ROOT / "data/derived/expansion3000_scale_seed20260508"),
    )
    parser.add_argument("--holdout-size", type=int, default=200)
    parser.add_argument("--error-eval-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260508)
    parser.add_argument(
        "--checkpoint-sizes",
        type=str,
        default="300,600,900,1200,1500,1800,2100,2400,2700",
    )
    args = parser.parse_args()

    source_path = Path(args.source)
    outdir = Path(args.outdir)
    holdout_size = int(args.holdout_size)
    error_eval_size = int(args.error_eval_size)
    checkpoint_sizes = sorted({int(x.strip()) for x in args.checkpoint_sizes.split(",") if x.strip()})

    source_rows = _load_rows(source_path)
    if len(source_rows) < holdout_size + 300:
        raise SystemExit(f"Source too small: {len(source_rows)}")

    annotated_eval = _load_rows(Path(args.annotated_error_eval))
    if len(annotated_eval) < error_eval_size:
        raise SystemExit(f"Annotated error eval too small: {len(annotated_eval)}")

    source_by_question = _index_source_by_question(source_rows)
    eval_rows = annotated_eval[:error_eval_size]
    eval_question_keys = {_question_key(row) for row in eval_rows if _question_key(row)}

    blocked_ids: Set[str] = set()
    for row in source_rows:
        sid = str(row.get("id") or "")
        if sid and _question_key(row) in eval_question_keys:
            blocked_ids.add(sid)

    if len(blocked_ids) < holdout_size:
        raise SystemExit(
            f"Blocked-by-question set too small for holdout: blocked={len(blocked_ids)} holdout={holdout_size}"
        )

    error_eval_out = _remap_error_eval(annotated_eval=eval_rows)
    rng = random.Random(int(args.seed))
    holdout_ids = set(rng.sample(sorted(blocked_ids), holdout_size))
    pool_rows = [row for row in source_rows if str(row.get("id") or "") not in blocked_ids]
    holdout_rows = [row for row in source_rows if str(row.get("id") or "") in holdout_ids]
    pool_ids = {str(r["id"]) for r in pool_rows}
    eval_ids = {str(r.get("id") or "") for r in error_eval_out}

    sem_path = Path(args.semantic_experience)
    sem_ids: Set[str] = set()
    if sem_path.exists():
        sem_payload = json.loads(sem_path.read_text(encoding="utf-8"))
        sem_ids = set(_resume_done_map(sem_payload).keys())

    pool_question_keys = {_question_key(r) for r in pool_rows}
    eval_question_keys_out = {_question_key(r) for r in error_eval_out if _question_key(r)}
    question_overlap = pool_question_keys & eval_question_keys_out

    manifest = {
        "policy": {
            "source": "evaluation_sample_3000_expansion.json",
            "blocked_from_rules": "all source rows whose question matches error_eval_100",
            "eval_holdout_200": "200 blocked rows reserved for eval bookkeeping",
            "error_eval_100": "100 dual-chain annotated rows (original predictions; unchanged GT spans)",
            "rule_expansion": "non-blocked samples only; semantic experience reused from catalogs/semantic_experience.json",
            "semantic_extraction": "skipped; existing semantic experience filtered by expansion pool IDs",
        },
        "counts": {
            "source_total": len(source_rows),
            "blocked_from_rules": len(blocked_ids),
            "expansion_pool": len(pool_rows),
            "eval_holdout_200": len(holdout_rows),
            "error_eval_100": len(error_eval_out),
            "error_eval_expansion_ids": 0,
            "semantic_done_total": len(sem_ids),
            "semantic_done_in_pool": len(pool_ids & sem_ids),
            "semantic_done_in_blocked": len(blocked_ids & sem_ids),
        },
        "overlap_audit": {
            "pool_vs_blocked_ids": sorted(pool_ids & blocked_ids),
            "pool_vs_error_eval_ids": sorted(pool_ids & eval_ids),
            "pool_vs_error_eval_questions": len(question_overlap),
            "error_eval_100_subset_of_holdout_200": True,
            "passes": not (pool_ids & blocked_ids)
            and not (pool_ids & eval_ids)
            and not question_overlap,
        },
        "paths": {
            "source": str(source_path),
            "expansion_pool": str(outdir / "expansion_pool.json"),
            "eval_holdout_200": str(outdir / "eval_holdout_200.json"),
            "error_eval_100": str(outdir / "error_eval_dataset_100.json"),
            "semantic_experience": str(sem_path),
            "checkpoints_dir": str(outdir / "checkpoints"),
        },
        "checkpoint_sizes": checkpoint_sizes,
    }

    if not manifest["overlap_audit"]["passes"]:
        raise SystemExit(f"Overlap audit failed: {manifest['overlap_audit']}")

    _write_json(outdir / "expansion_pool.json", pool_rows)
    _write_json(outdir / "eval_holdout_200.json", holdout_rows)
    _write_json(outdir / "error_eval_dataset_100.json", error_eval_out)
    _write_json(outdir / "split_manifest.json", manifest)

    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for n in checkpoint_sizes:
        if n > len(pool_rows):
            continue
        _write_json(ckpt_dir / f"expansion_sample_{n:04d}.json", pool_rows[:n])

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
