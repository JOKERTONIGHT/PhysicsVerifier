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

from training.compat.math_grading import grade_answer_verl
from training.rl_data.answer_equiv import answers_equivalent

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
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(x) for x in answer if x is not None and str(x).strip()]
    return [str(answer)]


def extract_all_boxed(text: str) -> List[str]:
    """Return inner contents of every ``\\boxed{}`` / ``\\fbox{}`` in order."""
    src = str(text or "")
    out: List[str] = []
    start = 0
    while start < len(src):
        idx_box = src.find("\\boxed", start)
        idx_fbox = src.find("\\fbox", start)
        candidates = [i for i in (idx_box, idx_fbox) if i >= 0]
        if not candidates:
            break
        idx = min(candidates)
        cmd_len = 6 if src.startswith("\\boxed", idx) else 5
        i = idx + cmd_len
        while i < len(src) and src[i].isspace():
            i += 1
        if i >= len(src):
            break
        if src[i] != "{":
            j = i
            while j < len(src) and (not src[j].isspace()) and src[j] not in "\\$":
                j += 1
            if j > i:
                out.append(src[i:j])
            start = max(j, i + 1)
            continue
        depth = 0
        j = i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(src[i + 1 : j])
                    start = j + 1
                    break
            j += 1
        else:
            break
    return out


def gold_parts(labels: Sequence[str]) -> List[str]:
    parts: List[str] = []
    for lab in labels:
        boxed = extract_all_boxed(lab)
        if boxed:
            parts.extend(p.strip() for p in boxed if str(p).strip())
            continue
        cleaned = str(lab).strip().strip("$").strip()
        if cleaned:
            parts.append(cleaned)
    return parts


def looks_repetitive(text: str, window: int = 200, min_repeats: int = 3) -> bool:
    src = str(text or "")
    if len(src) < window * min_repeats:
        return False
    needle = src[-window:]
    if not needle.strip():
        return False
    return src.count(needle) >= min_repeats


def boxed_unclosed(text: str) -> bool:
    src = str(text or "")
    idx = max(src.rfind("\\boxed"), src.rfind("\\fbox"))
    if idx < 0:
        return False
    i = idx
    while i < len(src) and src[i] not in "{":
        i += 1
    if i >= len(src) or src[i] != "{":
        return False
    depth = 0
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return False
        i += 1
    return True


def looks_truncated(text: str) -> bool:
    src = str(text or "").rstrip()
    if not src:
        return True
    if boxed_unclosed(src):
        return True
    if "\\boxed" not in src and "\\fbox" not in src:
        return not bool(END_PUNCT_RE.search(src))
    return False


def _as_boxed(text: str) -> str:
    src = str(text or "").strip()
    if "\\boxed" in src or "\\fbox" in src:
        return src if src.startswith("\\") else f"\\boxed{{{src}}}"
    return f"\\boxed{{{src}}}"


def answers_match(cand: str, gold: str) -> bool:
    if not cand or not gold:
        return False
    cand_box = _as_boxed(cand)
    gold_box = _as_boxed(gold)
    if grade_answer_verl(cand_box, gold_box):
        return True
    ok, _ = answers_equivalent(cand_box, gold_box)
    return bool(ok)


def score_prediction(pred: str, labels: Sequence[str]) -> Dict[str, Any]:
    pred = str(pred or "")
    boxes = extract_all_boxed(pred)
    parts = gold_parts(labels)
    n_parts = max(len(parts), 1) if parts else 0
    hits = 0
    for part in parts:
        if any(answers_match(box, part) for box in boxes):
            hits += 1
            continue
        if boxes and answers_match(pred, part):
            hits += 1
    item_correct = bool(parts) and hits == len(parts)
    part_frac = (hits / len(parts)) if parts else 0.0
    return {
        "n_parts": len(parts),
        "n_hit": hits,
        "part_frac": part_frac,
        "item_correct": item_correct,
        "n_boxed": len(boxes),
        "no_boxed": len(boxes) == 0,
        "truncated": looks_truncated(pred),
        "repetitive": looks_repetitive(pred),
    }


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
    part_avgs: List[float] = []
    item_avgs: List[float] = []
    item_pass: List[float] = []
    part_pass: List[float] = []
    for recs in groups.values():
        part_avgs.append(sum(r["part_frac"] for r in recs) / max(len(recs), 1))
        item_avgs.append(sum(float(r["item_correct"]) for r in recs) / max(len(recs), 1))
        item_pass.append(1.0 if any(r["item_correct"] for r in recs) else 0.0)
        part_pass.append(max((r["part_frac"] for r in recs), default=0.0))
    part_avg_at_k = sum(part_avgs) / n_items
    item_avg_at_k = sum(item_avgs) / n_items
    item_pass_at_k = sum(item_pass) / n_items
    part_pass_at_k = sum(part_pass) / n_items
    se = math.sqrt(part_avg_at_k * (1.0 - part_avg_at_k) / n_items) if 0.0 < part_avg_at_k < 1.0 else 0.0

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
        "part_avg_at_k_se": se,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
