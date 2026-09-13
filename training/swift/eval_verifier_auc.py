#!/usr/bin/env python3
"""Score process-reward ranking quality vs answer correctness.

Reads labelled rollouts jsonl with fields:
  sample_id, prediction/response, label/solution, part_frac (optional)
and a sidecar of process scores or computes them from paragraph_process
given diagnostics. For the verifier path, pass --verifier-json produced by
scripts/run_verifier.py (list of {question, prediction, diagnostics, ...}).

Gate: ROC-AUC of process score vs item_correct >= --min-auc (default 0.65).
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.compat.part_scoring import score_prediction
from training.reward_server.paragraph_process import score_text_with_diagnostics


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _labels(row: Dict[str, Any]) -> List[str]:
    lab = row.get("label") if row.get("label") is not None else row.get("solution")
    if lab is None:
        lab = row.get("answer")
    if isinstance(lab, list):
        return [str(x) for x in lab]
    if lab is None:
        return []
    return [str(lab)]


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pairs = list(zip(scores, labels))
    pos = [s for s, y in pairs if y == 1]
    neg = [s for s, y in pairs if y == 0]
    if not pos or not neg:
        return float("nan")
    gt = 0.0
    eq = 0.0
    for s in pos:
        for t in neg:
            if s > t:
                gt += 1
            elif s == t:
                eq += 1
    return (gt + 0.5 * eq) / (len(pos) * len(neg))


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    hits = 0
    prec_sum = 0.0
    n_pos = sum(labels)
    if n_pos <= 0:
        return float("nan")
    for rank, i in enumerate(order, start=1):
        if labels[i] == 1:
            hits += 1
            prec_sum += hits / rank
    return prec_sum / n_pos


def point_biserial(scores: Sequence[float], labels: Sequence[int]) -> float:
    n = len(scores)
    if n < 3:
        return float("nan")
    mean = sum(scores) / n
    var = sum((s - mean) ** 2 for s in scores) / n
    if var <= 0:
        return 0.0
    sd = math.sqrt(var)
    n1 = sum(labels)
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    m1 = sum(s for s, y in zip(scores, labels) if y == 1) / n1
    m0 = sum(s for s, y in zip(scores, labels) if y == 0) / n0
    return ((m1 - m0) / sd) * math.sqrt(n1 * n0 / (n * n))


def process_from_row(row: Dict[str, Any]) -> float:
    if row.get("process_score") is not None:
        return float(row["process_score"])
    diags = row.get("diagnostics") or []
    pred = str(row.get("prediction") or row.get("response") or row.get("completion") or "")
    scored = score_text_with_diagnostics(pred, diags)
    return float(scored.get("score") or 0.0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollouts", type=Path, required=True)
    p.add_argument("--verifier-json", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-auc", type=float, default=0.65)
    args = p.parse_args()

    rows = _load_jsonl(args.rollouts)
    verifier_by_key: Dict[str, Dict[str, Any]] = {}
    if args.verifier_json and args.verifier_json.is_file():
        blob = json.loads(args.verifier_json.read_text(encoding="utf-8"))
        items = blob if isinstance(blob, list) else blob.get("results") or blob.get("samples") or []
        for item in items:
            if not isinstance(item, dict):
                continue
            key = str(item.get("prediction") or item.get("response") or "")[:400]
            verifier_by_key[key] = item

    scores: List[float] = []
    labels: List[int] = []
    part_fracs: List[float] = []
    groups: Dict[str, List[Tuple[float, int]]] = defaultdict(list)
    for row in rows:
        pred = str(row.get("prediction") or row.get("response") or row.get("completion") or "")
        if args.verifier_json:
            hit = verifier_by_key.get(pred[:400], {})
            if hit.get("diagnostics") is not None:
                row = dict(row)
                row["diagnostics"] = hit.get("diagnostics") or []
        gold = _labels(row)
        metrics = score_prediction(pred, gold) if gold else {
            "item_correct": bool(row.get("item_correct") or row.get("acc")),
            "part_frac": float(row.get("part_frac") or 0.0),
        }
        y = 1 if metrics["item_correct"] else 0
        s = process_from_row(row)
        scores.append(s)
        labels.append(y)
        part_fracs.append(float(metrics.get("part_frac") or 0.0))
        sid = str(row.get("sample_id") or (row.get("metadata") or {}).get("sample_id") or "")
        groups[sid].append((s, y))

    n_effective = sum(1 for recs in groups.values() if recs and max(r[0] for r in recs) - min(r[0] for r in recs) > 1e-6)
    report = {
        "n": len(scores),
        "n_pos": int(sum(labels)),
        "n_neg": int(len(labels) - sum(labels)),
        "process_mean": (sum(scores) / len(scores)) if scores else 0.0,
        "auc": roc_auc(scores, labels),
        "ap": average_precision(scores, labels),
        "point_biserial": point_biserial(scores, labels),
        "effective_group_rate": n_effective / max(len(groups), 1),
        "n_groups": len(groups),
        "gate_min_auc": args.min_auc,
        "pass": False,
    }
    auc = report["auc"]
    report["pass"] = bool(auc == auc and auc >= args.min_auc)
    if auc == auc and auc >= 0.8:
        report["recommended_w_process"] = 0.4
    elif auc == auc and auc >= args.min_auc:
        report["recommended_w_process"] = 0.2
    else:
        report["recommended_w_process"] = 0.0
        report["do_not_use_process_reward"] = True
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
