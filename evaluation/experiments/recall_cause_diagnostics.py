#!/usr/bin/env python3
"""Forensic analysis of low recall across rule-based and pure-LLM checker experiments."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    if not quote:
        ev = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
        quote = str(ev.get("quote") or "").strip()
    return f"{msg} {quote}".strip()


def _finding_para(d: Dict[str, Any]) -> int:
    if "paragraph_index" in d:
        return int(d.get("paragraph_index") or -1)
    ev = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
    loc = ev.get("location") if isinstance(ev.get("location"), dict) else {}
    return int(loc.get("paragraph_index") or -1)


def _gt_para(g: Dict[str, Any]) -> int:
    return int(g.get("paragraph_index") or -1)


def classify_unmatched_gt(
    gt: Dict[str, Any],
    findings: List[Dict[str, Any]],
    *,
    location_matched: bool,
) -> Tuple[str, Dict[str, Any]]:
    if location_matched:
        return "matched", {}
    gt_text = str(gt.get("error_text") or "")
    gt_para = _gt_para(gt)
    if not findings:
        return "no_detection", {"gt_paragraph_index": gt_para}

    best_overlap = 0
    best_pred: Optional[Dict[str, Any]] = None
    same_para_preds = 0
    for f in findings:
        ft = _finding_text(f)
        ov = _overlap(gt_text, ft)
        if ov > best_overlap:
            best_overlap = ov
            best_pred = f
        if _finding_para(f) == gt_para and gt_para >= 1:
            same_para_preds += 1

    meta: Dict[str, Any] = {
        "best_token_overlap": best_overlap,
        "same_paragraph_preds": same_para_preds,
        "gt_paragraph_index": gt_para,
    }
    if best_pred:
        meta["best_pred_preview"] = _finding_text(best_pred)[:180]

    if best_overlap >= 4 and same_para_preds > 0:
        return "location_failure", meta
    if best_overlap >= 3:
        return "semantic_near_miss", meta
    if same_para_preds > 0 and best_overlap >= 1:
        return "adjacent_concept", meta
    return "semantic_gap", meta


def analyze_experiment(
    *,
    label: str,
    dataset_rows: List[Dict[str, Any]],
    results_rows: List[Dict[str, Any]],
    iou_threshold: float,
    coverage_threshold: float,
) -> Dict[str, Any]:
    ds_by_id = _index_by_id(dataset_rows)
    pred_by_id = _index_by_id(results_rows)
    unmatched_records: List[Dict[str, Any]] = []
    cause_counter: Counter = Counter()
    per_sample: List[Dict[str, Any]] = []

    for sid, row in ds_by_id.items():
        gt_entries = _extract_gt_entries(row, sid)
        loc_gt = [g for g in gt_entries if bool(g.get("locatable_valid"))]
        pred_item = pred_by_id.get(sid, {})
        findings = _collect_pred_findings(pred_item, pred_item)
        findings = _fill_missing_pred_locations(findings, answer_text=str(row.get("prediction") or ""))

        matched_ids: Set[str] = set()
        if loc_gt:
            matched_ids, _, _ = _match_by_location(
                loc_gt, findings, iou_threshold=iou_threshold, coverage_threshold=coverage_threshold
            )

        sample_causes: Counter = Counter()
        for g in gt_entries:
            eid = str(g.get("error_id") or "")
            cause, meta = classify_unmatched_gt(g, findings, location_matched=eid in matched_ids)
            if cause != "matched":
                cause_counter[cause] += 1
                sample_causes[cause] += 1
                unmatched_records.append(
                    {
                        "experiment": label,
                        "sample_id": sid,
                        "error_id": eid,
                        "cause": cause,
                        "error_preview": str(g.get("error_text") or "")[:200],
                        **meta,
                    }
                )
        per_sample.append(
            {
                "sample_id": sid,
                "gt_count": len(gt_entries),
                "matched_count": len(matched_ids),
                "unmatched_causes": dict(sample_causes),
            }
        )

    return {
        "label": label,
        "unmatched_gt_causes": dict(cause_counter),
        "unmatched_records": unmatched_records,
        "per_sample": per_sample,
    }


def common_missed_gt(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_exp: Dict[str, Set[str]] = {}
    detail_by_key: Dict[str, Dict[str, Any]] = {}
    for a in analyses:
        label = str(a.get("label") or "")
        missed: Set[str] = set()
        for rec in a.get("unmatched_records") or []:
            if not isinstance(rec, dict):
                continue
            key = f"{rec.get('sample_id')}::{rec.get('error_id')}"
            missed.add(key)
            detail_by_key[key] = rec
        by_exp[label] = missed

    if not by_exp:
        return {"common_to_all": [], "by_experiment": {}}

    all_keys = set.intersection(*by_exp.values()) if by_exp else set()
    union_keys = set.union(*by_exp.values()) if by_exp else set()
    common_items = []
    for key in sorted(all_keys):
        rec = detail_by_key.get(key, {})
        common_items.append(
            {
                "key": key,
                "sample_id": rec.get("sample_id"),
                "error_id": rec.get("error_id"),
                "error_preview": rec.get("error_preview"),
            }
        )
    return {
        "common_to_all_experiments": common_items,
        "common_count": len(common_items),
        "union_missed_count": len(union_keys),
        "by_experiment_missed_count": {k: len(v) for k, v in by_exp.items()},
    }


def checker_fp_report(
    *,
    label: str,
    dataset_rows: List[Dict[str, Any]],
    results_rows: List[Dict[str, Any]],
    iou_threshold: float,
    coverage_threshold: float,
) -> List[Dict[str, Any]]:
    ds_by_id = _index_by_id(dataset_rows)
    pred_by_id = _index_by_id(results_rows)
    fps: List[Dict[str, Any]] = []
    for sid, row in ds_by_id.items():
        gt_entries = _extract_gt_entries(row, sid)
        loc_gt = [g for g in gt_entries if bool(g.get("locatable_valid"))]
        pred_item = pred_by_id.get(sid, {})
        findings = _collect_pred_findings(pred_item, pred_item)
        findings = _fill_missing_pred_locations(findings, answer_text=str(row.get("prediction") or ""))
        _, _, used = _match_by_location(
            loc_gt, findings, iou_threshold=iou_threshold, coverage_threshold=coverage_threshold
        ) if loc_gt else (set(), [], set())
        for idx, f in enumerate(findings):
            if idx in used:
                continue
            if not bool(f.get("locatable_valid")):
                continue
            ft = _finding_text(f)
            best_ov = max((_overlap(ft, str(g.get("error_text") or "")) for g in gt_entries), default=0)
            fps.append(
                {
                    "experiment": label,
                    "sample_id": sid,
                    "message_preview": ft[:200],
                    "best_gt_token_overlap": best_ov,
                    "likely_missing_gt": best_ov >= 3,
                }
            )
    return fps


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall cause diagnostics across experiments.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--experiment",
        action="append",
        nargs=2,
        metavar=("LABEL", "RESULTS_JSON"),
        default=[],
        help="Repeatable: label path/to/results.json",
    )
    parser.add_argument("--location-iou-threshold", type=float, default=0.5)
    parser.add_argument("--location-coverage-threshold", type=float, default=0.6)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_rows = _load_json(args.dataset)
    if not isinstance(dataset_rows, list):
        raise SystemExit("dataset must be a JSON array")

    experiments = list(args.experiment or [])
    if not experiments:
        raise SystemExit("Provide at least one --experiment LABEL RESULTS_JSON")

    analyses: List[Dict[str, Any]] = []
    all_fps: List[Dict[str, Any]] = []
    for label, results_path in experiments:
        results_rows = _load_json(str(results_path))
        if not isinstance(results_rows, list):
            raise SystemExit(f"{results_path} must be a JSON array")
        analyses.append(
            analyze_experiment(
                label=str(label),
                dataset_rows=dataset_rows,
                results_rows=results_rows,
                iou_threshold=float(args.location_iou_threshold),
                coverage_threshold=float(args.location_coverage_threshold),
            )
        )
        all_fps.extend(
            checker_fp_report(
                label=str(label),
                dataset_rows=dataset_rows,
                results_rows=results_rows,
                iou_threshold=float(args.location_iou_threshold),
                coverage_threshold=float(args.location_coverage_threshold),
            )
        )

    common = common_missed_gt(analyses)
    by_cause: Dict[str, Counter] = defaultdict(Counter)
    for a in analyses:
        for cause, cnt in (a.get("unmatched_gt_causes") or {}).items():
            by_cause[cause][str(a.get("label"))] += int(cnt)

    forensics = {
        "dataset": args.dataset,
        "experiments": [
            {
                "label": a["label"],
                "unmatched_gt_causes": a["unmatched_gt_causes"],
                "unmatched_count": sum((a.get("unmatched_gt_causes") or {}).values()),
            }
            for a in analyses
        ],
        "aggregate_unmatched_causes": {k: dict(v) for k, v in by_cause.items()},
        "common_missed_gt": common,
        "likely_missing_gt_fp_count": sum(1 for x in all_fps if x.get("likely_missing_gt")),
        "total_fp_count": len(all_fps),
    }

    (out_dir / "recall_failure_forensics.json").write_text(
        json.dumps(forensics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "unmatched_gt_by_cause.json").write_text(
        json.dumps(
            [r for a in analyses for r in (a.get("unmatched_records") or [])],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "checker_fp_or_missed_label_report.json").write_text(
        json.dumps(all_fps, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "common_missed_gt_report.json").write_text(
        json.dumps(common, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(forensics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
