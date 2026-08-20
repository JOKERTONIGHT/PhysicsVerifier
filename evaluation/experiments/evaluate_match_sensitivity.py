#!/usr/bin/env python3
"""Multi-layer matching sensitivity analysis for recall attribution."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from evaluate_physics_eval_sets import (  # type: ignore
    _collect_pred_findings,
    _extract_gt_entries,
    _fill_missing_pred_locations,
    _index_by_id,
    _load_json,
    _match_by_location,
)


def _tokenize(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(t) >= 3}


def _overlap(a: str, b: str) -> int:
    return len(_tokenize(a) & _tokenize(b))


def _finding_text(d: Dict[str, Any]) -> str:
    msg = str(d.get("message") or d.get("text") or "").strip()
    quote = str(d.get("quote") or "").strip()
    return f"{msg} {quote}".strip()


def _infer_gt_groups(gt_entries: List[Dict[str, Any]]) -> Dict[str, str]:
    """Assign group_id by token overlap clustering within a sample."""
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ids = [str(g.get("error_id") or "") for g in gt_entries if str(g.get("error_id") or "")]
    texts = {str(g.get("error_id") or ""): str(g.get("error_text") or "") for g in gt_entries}
    paras = {str(g.get("error_id") or ""): int(g.get("paragraph_index") or -1) for g in gt_entries}
    for i, a in enumerate(ids):
        for b in ids[i + 1 :]:
            if paras.get(a, -1) == paras.get(b, -1) and paras.get(a, -1) >= 1:
                if _overlap(texts[a], texts[b]) >= 2:
                    union(a, b)
            elif _overlap(texts[a], texts[b]) >= 4:
                union(a, b)
    groups: Dict[str, str] = {}
    for eid in ids:
        groups[eid] = find(eid)
    return groups


def _semantic_detection_match(gt_text: str, findings: List[Dict[str, Any]], *, threshold: int = 3) -> bool:
    for f in findings:
        if _overlap(gt_text, _finding_text(f)) >= threshold:
            return True
    return False


def _paragraph_match(gt: Dict[str, Any], findings: List[Dict[str, Any]]) -> bool:
    g_para = int(gt.get("paragraph_index") or -1)
    if g_para < 1:
        return False
    gt_text = str(gt.get("error_text") or "")
    for f in findings:
        ev = f if "paragraph_index" in f else {}
        p_para = int(ev.get("paragraph_index") or -1)
        if p_para == g_para and _overlap(gt_text, _finding_text(f)) >= 2:
            return True
    return False


def evaluate_layers(
    *,
    dataset_rows: List[Dict[str, Any]],
    results_rows: List[Dict[str, Any]],
    strict_iou: float,
    strict_cov: float,
    relaxed_iou: float,
    relaxed_cov: float,
) -> Dict[str, Any]:
    ds_by_id = _index_by_id(dataset_rows)
    pred_by_id = _index_by_id(results_rows)

    total_gt = 0
    strict_hits = 0
    relaxed_hits = 0
    paragraph_hits = 0
    semantic_hits = 0
    group_hits = 0
    group_total = 0

    for sid, row in ds_by_id.items():
        gt_entries = _extract_gt_entries(row, sid)
        loc_gt = [g for g in gt_entries if bool(g.get("locatable_valid"))]
        pred_item = pred_by_id.get(sid, {})
        findings = _collect_pred_findings(pred_item, pred_item)
        findings = _fill_missing_pred_locations(findings, answer_text=str(row.get("prediction") or ""))

        groups = _infer_gt_groups(gt_entries)
        group_to_gt: Dict[str, List[str]] = defaultdict(list)
        for g in gt_entries:
            eid = str(g.get("error_id") or "")
            group_to_gt[groups.get(eid, eid)].append(eid)

        strict_ids: Set[str] = set()
        relaxed_ids: Set[str] = set()
        if loc_gt:
            strict_ids, _, _ = _match_by_location(loc_gt, findings, iou_threshold=strict_iou, coverage_threshold=strict_cov)
            relaxed_ids, _, _ = _match_by_location(loc_gt, findings, iou_threshold=relaxed_iou, coverage_threshold=relaxed_cov)

        for g in gt_entries:
            total_gt += 1
            eid = str(g.get("error_id") or "")
            if eid in strict_ids:
                strict_hits += 1
            if eid in relaxed_ids:
                relaxed_hits += 1
            if _paragraph_match(g, findings):
                paragraph_hits += 1
            if _semantic_detection_match(str(g.get("error_text") or ""), findings):
                semantic_hits += 1

        for _gid, members in group_to_gt.items():
            group_total += 1
            if any(
                eid in strict_ids
                or _paragraph_match(next(x for x in gt_entries if str(x.get("error_id")) == eid), findings)
                or _semantic_detection_match(
                    str(next(x for x in gt_entries if str(x.get("error_id")) == eid).get("error_text") or ""),
                    findings,
                )
                for eid in members
            ):
                group_hits += 1

    def _recall(h: int, t: int) -> float:
        return round(h / max(1, t), 4)

    return {
        "total_gt_errors": total_gt,
        "strict_span_recall": _recall(strict_hits, total_gt),
        "relaxed_span_recall": _recall(relaxed_hits, total_gt),
        "paragraph_recall": _recall(paragraph_hits, total_gt),
        "semantic_detection_recall": _recall(semantic_hits, total_gt),
        "group_recall": _recall(group_hits, group_total),
        "strict_hits": strict_hits,
        "relaxed_hits": relaxed_hits,
        "paragraph_hits": paragraph_hits,
        "semantic_hits": semantic_hits,
        "group_hits": group_hits,
        "group_total": group_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-layer match sensitivity evaluation.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--strict-iou", type=float, default=0.5)
    parser.add_argument("--strict-coverage", type=float, default=0.6)
    parser.add_argument("--relaxed-iou", type=float, default=0.2)
    parser.add_argument("--relaxed-coverage", type=float, default=0.3)
    args = parser.parse_args()

    dataset_rows = _load_json(args.dataset)
    results_rows = _load_json(args.results)
    if not isinstance(dataset_rows, list) or not isinstance(results_rows, list):
        raise SystemExit("dataset and results must be JSON arrays")

    layers = evaluate_layers(
        dataset_rows=dataset_rows,
        results_rows=results_rows,
        strict_iou=float(args.strict_iou),
        strict_cov=float(args.strict_coverage),
        relaxed_iou=float(args.relaxed_iou),
        relaxed_cov=float(args.relaxed_coverage),
    )
    out = {"label": args.label, "dataset": args.dataset, "results": args.results, **layers}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
