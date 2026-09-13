#!/usr/bin/env python3
"""Score HiPhO / heldout predictions with part-level answer metrics.

Legacy ``boxed_acc`` / ``answer_acc`` equal item-level exactness (all gold
parts hit). New fields add part-level accuracy, degradation diagnostics, and
avg@k / pass@k when ``sample_index`` is present or ``--n-samples`` is set.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from training.compat.part_scoring import (
    answers_match,
    bootstrap_mean_ci,
    boxed_unclosed,
    extract_all_boxed,
    gold_parts,
    group_item_scores,
    labels_from_value,
    looks_repetitive,
    looks_truncated,
    paired_delta_ci,
    score_prediction,
)

DEFAULT_UNIFIED_RULES = ROOT / "catalogs/rules_unified_3000_runtime_backfilled.json"
END_PUNCT_RE = re.compile(r"[.。!?）)\}\]$]$")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _labels(answer: Any) -> List[str]:
    return labels_from_value(answer)


def item_key(row: Dict[str, Any]) -> str:
    sid = row.get("sample_id") or (row.get("metadata") or {}).get("sample_id") or row.get("id")
    if sid:
        return f"id:{sid}"
    q = row.get("question") or (row.get("metadata") or {}).get("question") or ""
    if q:
        return "q:" + str(q)[:400]
    inp = row.get("input")
    if isinstance(inp, list) and inp:
        last = inp[-1] if isinstance(inp[-1], dict) else {}
        return "u:" + str(last.get("content") or "")[:400]
    return f"row:{id(row)}"


def _row_labels(row: Dict[str, Any], gold_by_id: Dict[str, Dict[str, Any]]) -> List[str]:
    labels = _labels(row.get("answer") if row.get("answer") is not None else row.get("label"))
    if labels:
        return labels
    gid = str(row.get("id") or row.get("sample_id") or (row.get("metadata") or {}).get("sample_id") or "")
    gold = gold_by_id.get(gid, {})
    return _labels(gold.get("answer") if gold.get("answer") is not None else gold.get("label"))


def summarize_scores(
    rows: Sequence[Dict[str, Any]],
    *,
    gold_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    verifier: Any = None,
    k_hint: int = 0,
) -> Dict[str, Any]:
    gold_by_id = gold_by_id or {}
    per_exam: Dict[str, Dict[str, float]] = {}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_errors = 0
    n_parts_all = 0
    n_hit_all = 0
    n_item_hit = 0
    n_no_boxed = 0
    n_trunc = 0
    n_rep = 0

    scored_rows: List[Dict[str, Any]] = []
    for row in rows:
        pred = str(row.get("prediction") or row.get("response") or row.get("completion") or "")
        labels = _row_labels(row, gold_by_id)
        metrics = score_prediction(pred, labels)
        n_parts_all += metrics["n_parts"]
        n_hit_all += metrics["n_hit"]
        n_item_hit += int(metrics["item_correct"])
        n_no_boxed += int(metrics["no_boxed"])
        n_trunc += int(metrics["truncated"])
        n_rep += int(metrics["repetitive"])

        n_errors = 0
        if verifier is not None:
            result = verifier.verify({"question": row.get("question", ""), "prediction": pred})
            n_errors = sum(
                1 for d in (result.get("diagnostics") or []) if str(d.get("severity", "")).lower() == "error"
            )
        total_errors += n_errors

        exam = str((row.get("metadata") or {}).get("exam") or "unknown")
        bucket = per_exam.setdefault(exam, {"n": 0, "acc": 0, "part_hit": 0, "n_parts": 0, "errors": 0})
        bucket["n"] += 1
        bucket["acc"] += int(metrics["item_correct"])
        bucket["part_hit"] += metrics["n_hit"]
        bucket["n_parts"] += metrics["n_parts"]
        bucket["errors"] += n_errors

        rec = dict(metrics)
        rec["exam"] = exam
        rec["sample_index"] = row.get("sample_index", 0)
        groups[item_key(row)].append(rec)
        scored_rows.append(rec)

    n_rows = max(len(rows), 1)
    item_acc = n_item_hit / n_rows
    part_acc = (n_hit_all / n_parts_all) if n_parts_all else 0.0
    no_boxed_rate = n_no_boxed / n_rows
    truncated_rate = n_trunc / n_rows
    repetition_rate = n_rep / n_rows
    degrade_rate = no_boxed_rate + repetition_rate

    sizes = [len(v) for v in groups.values()]
    k = k_hint or (max(sizes) if sizes else 1)
    n_items = max(len(groups), 1)
    part_avgs, item_avgs, item_pass, part_pass = group_item_scores(groups)
    part_avg_at_k = sum(part_avgs) / n_items
    item_avg_at_k = sum(item_avgs) / n_items
    item_pass_at_k = sum(item_pass) / n_items
    part_pass_at_k = sum(part_pass) / n_items
    se = math.sqrt(part_avg_at_k * (1.0 - part_avg_at_k) / n_items) if 0.0 < part_avg_at_k < 1.0 else 0.0
    part_ci = bootstrap_mean_ci(part_avgs)
    convergence_gap = part_pass_at_k - part_avg_at_k

    summary = {
        "n_samples": len(groups),
        "n_rows": len(rows),
        "k": int(k),
        "boxed_acc": item_acc,
        "answer_acc": item_acc,
        "item_acc": item_acc,
        "part_acc": part_acc,
        "part_avg_at_k": part_avg_at_k,
        "item_avg_at_k": item_avg_at_k,
        "item_pass_at_k": item_pass_at_k,
        "part_pass_at_k": part_pass_at_k,
        "part_pass_minus_avg": convergence_gap,
        "part_avg_at_k_se": se,
        "part_avg_at_k_ci": part_ci,
        "part_avg_item_scores": part_avgs,
        "no_boxed_rate": no_boxed_rate,
        "truncated_rate": truncated_rate,
        "repetition_rate": repetition_rate,
        "degrade_rate": degrade_rate,
        "metric_note": (
            "item_acc is all-parts-hit; part_acc is hit-parts/total-parts; "
            "boxed_acc/answer_acc alias item_acc. Not official HiPhO."
        ),
        "avg_process_errors": total_errors / n_rows,
        "per_exam": {
            name: {
                "n": int(v["n"]),
                "boxed_acc": v["acc"] / max(v["n"], 1),
                "answer_acc": v["acc"] / max(v["n"], 1),
                "item_acc": v["acc"] / max(v["n"], 1),
                "part_acc": v["part_hit"] / max(v["n_parts"], 1),
                "avg_process_errors": v["errors"] / max(v["n"], 1),
            }
            for name, v in per_exam.items()
        },
    }
    return summary


def evaluate_gate(
    sft: Dict[str, Any],
    base: Dict[str, Any],
    *,
    min_part_avg: float | None = None,
    max_degrade_rate: float | None = None,
    max_no_boxed_rate: float | None = None,
) -> Dict[str, Any]:
    sft_part = float(sft.get("part_avg_at_k") if sft.get("part_avg_at_k") is not None else sft.get("part_acc") or 0.0)
    base_part = float(base.get("part_avg_at_k") if base.get("part_avg_at_k") is not None else base.get("part_acc") or 0.0)
    n = int(base.get("n_samples") or 1)
    se = float(base.get("part_avg_at_k_se") or 0.0)
    if se == 0.0 and 0.0 < base_part < 1.0:
        se = math.sqrt(base_part * (1.0 - base_part) / max(n, 1))
    sft_deg = float(sft.get("degrade_rate") or 0.0)
    base_deg = float(base.get("degrade_rate") or 0.0)
    sft_no_box = float(sft.get("no_boxed_rate") or 0.0)
    passed = (sft_part >= base_part - se) and (sft_deg <= base_deg + 1e-12)
    extra_fail: List[str] = []
    if min_part_avg is not None and sft_part < float(min_part_avg):
        passed = False
        extra_fail.append("min_part_avg")
    if max_degrade_rate is not None and sft_deg > float(max_degrade_rate):
        passed = False
        extra_fail.append("max_degrade_rate")
    if max_no_boxed_rate is not None and sft_no_box > float(max_no_boxed_rate):
        passed = False
        extra_fail.append("max_no_boxed_rate")
    return {
        "sft_part_avg_at_k": sft_part,
        "base_part_avg_at_k": base_part,
        "delta_part": sft_part - base_part,
        "se": se,
        "sft_degrade_rate": sft_deg,
        "base_degrade_rate": base_deg,
        "sft_item_acc": float(sft.get("item_acc") or sft.get("answer_acc") or 0.0),
        "base_item_acc": float(base.get("item_acc") or base.get("answer_acc") or 0.0),
        "sft_no_boxed_rate": sft_no_box,
        "base_no_boxed_rate": float(base.get("no_boxed_rate") or 0.0),
        "sft_repetition_rate": float(sft.get("repetition_rate") or 0.0),
        "base_repetition_rate": float(base.get("repetition_rate") or 0.0),
        "pass": passed,
        "extra_fail": extra_fail,
        "rule": "sft_part_avg_at_k >= base_part_avg_at_k - 1*se AND sft_degrade_rate <= base_degrade_rate",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=None, help="Optional gold jsonl if predictions were stripped")
    parser.add_argument("--n-samples", type=int, default=0, help="Hint for k; auto-detected from sample_index if unset")
    parser.add_argument("--use-verifier", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Optional baseline scores.json with part_avg_item_scores for paired comparison",
    )
    args = parser.parse_args()

    rows = _load_jsonl(args.predictions)
    gold_by_id: Dict[str, Dict[str, Any]] = {}
    if args.gold:
        for grow in _load_jsonl(args.gold):
            gid = str(grow.get("id") or grow.get("sample_id") or (grow.get("metadata") or {}).get("sample_id") or "")
            if gid:
                gold_by_id[gid] = grow
    verifier = None
    if args.use_verifier:
        from core.physics_rule_verifier import PhysicsRuleVerifier

        symbolic_enabled = os.environ.get("PHYSICSVERIFIER_SYMBOLIC_ENABLED", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        verifier = PhysicsRuleVerifier(
            llm_model=os.environ.get(
                "PHYSICSVERIFIER_LLM_MODEL",
                "qwen3-30b-a3b-instruct-2507",
            ),
            unified_rules_path=os.environ.get(
                "PHYSICSVERIFIER_UNIFIED_RULES",
                str(DEFAULT_UNIFIED_RULES),
            ),
            enable_symbolic_check=symbolic_enabled,
            unified_retrieval_mode=os.environ.get(
                "PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE",
                "semantic",
            ),
            semantic_output_adapter=os.environ.get("PHYSICSVERIFIER_SEMANTIC_OUTPUT_ADAPTER") or None,
        )

    summary = summarize_scores(rows, gold_by_id=gold_by_id, verifier=verifier, k_hint=args.n_samples)
    summary["predictions"] = str(args.predictions)
    if args.baseline and args.baseline.is_file():
        base = json.loads(args.baseline.read_text(encoding="utf-8"))
        a = summary.get("part_avg_item_scores") or []
        b = base.get("part_avg_item_scores") or []
        if a and b and len(a) == len(b):
            summary["paired_vs_baseline"] = paired_delta_ci(a, b)
        else:
            summary["paired_vs_baseline"] = {
                "error": "item-score length mismatch",
                "n_this": len(a),
                "n_baseline": len(b),
            }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
