#!/usr/bin/env python3
"""Forensic annotation audit focused on recall attribution priority samples."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from audit_annotation_reasonableness import (  # type: ignore
    _audit_sample,
    _openai_client,
)
from dotenv import load_dotenv


FORENSIC_LABELS = {
    "valid",
    "valid_root_error",
    "valid_but_too_fine",
    "consequence_only",
    "duplicate",
    "duplicate_root_cause",
    "quote_misaligned",
    "questionable",
    "not_error",
    "over_granular",
}


def _map_forensic_label(raw: str, quote_grounded: bool) -> str:
    label = str(raw or "").strip().lower()
    if label == "valid":
        return "valid_root_error" if quote_grounded else "quote_misaligned"
    if label == "duplicate":
        return "duplicate_root_cause"
    if label == "over_granular":
        return "valid_but_too_fine"
    if label in FORENSIC_LABELS:
        return label
    return "questionable"


def _map_forensic_from_existing_item(item: Dict[str, Any]) -> str:
    label = str(item.get("label") or "").strip().lower()
    quote_grounded = bool(item.get("quote_grounded"))
    return _map_forensic_label(label, quote_grounded)


def _build_from_existing_audit(audit: Dict[str, Any], *, fp_report_path: str = "") -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    summary_counts: Counter = Counter()
    total_items = 0
    for row in audit.get("details") or []:
        if not isinstance(row, dict):
            continue
        mapped_items = []
        label_counter: Counter = Counter()
        for item in row.get("items") or []:
            if not isinstance(item, dict):
                continue
            forensic = _map_forensic_from_existing_item(item)
            mapped_items.append({**item, "forensic_label": forensic})
            label_counter[forensic] += 1
            summary_counts[forensic] += 1
            total_items += 1
        details.append(
            {
                "id": str(row.get("id") or ""),
                "audit_ok": bool(row.get("audit_ok")),
                "audit_error": row.get("error"),
                "gt_count": int(row.get("gt_count") or len(mapped_items)),
                "forensic_label_counts": dict(label_counter),
                "items": mapped_items,
            }
        )
    problematic = (
        summary_counts.get("not_error", 0)
        + summary_counts.get("duplicate_root_cause", 0)
        + summary_counts.get("consequence_only", 0)
        + summary_counts.get("valid_but_too_fine", 0)
    )
    quote_misaligned = summary_counts.get("quote_misaligned", 0)
    revision = _build_revision_candidates(details)
    fp_missing = 0
    if Path(str(fp_report_path or "")).exists():
        fps = _load_json(str(fp_report_path))
        if isinstance(fps, list):
            fp_missing = sum(1 for row in fps if isinstance(row, dict) and row.get("likely_missing_gt"))
    return {
        "summary": {
            "model": (audit.get("summary") or {}).get("model", "existing_audit"),
            "samples_audited": len(details),
            "total_gt_items": total_items,
            "forensic_label_counts": dict(summary_counts),
            "problematic_ratio": round(problematic / max(1, total_items), 4),
            "quote_misaligned_ratio": round(quote_misaligned / max(1, total_items), 4),
            "valid_root_error_ratio": round(summary_counts.get("valid_root_error", 0) / max(1, total_items), 4),
            "missing_gt_candidates_from_fp": fp_missing,
            "source": "existing_audit_mapping",
        },
        "details": details,
        "revision_candidates": revision,
    }


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _priority_sample_ids(
    *,
    dataset_rows: List[Dict[str, Any]],
    forensics_path: str,
    fp_report_path: str,
    top_n: int,
) -> Set[str]:
    wanted: Set[str] = set()
    if Path(forensics_path).exists():
        common = _load_json(forensics_path)
        for item in common.get("common_to_all_experiments") or []:
            if isinstance(item, dict) and item.get("sample_id"):
                wanted.add(str(item["sample_id"]))

    if Path(fp_report_path).exists():
        fps = _load_json(fp_report_path)
        if isinstance(fps, list):
            by_sample: Counter = Counter()
            for row in fps:
                if isinstance(row, dict) and row.get("likely_missing_gt"):
                    by_sample[str(row.get("sample_id") or "")] += 1
            for sid, _ in by_sample.most_common(top_n):
                if sid:
                    wanted.add(sid)

    gt_counts = []
    for row in dataset_rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "")
        gt_n = len(row.get("physics_error_gt") or [])
        gt_counts.append((gt_n, sid))
    gt_counts.sort(reverse=True)
    for _, sid in gt_counts[:top_n]:
        wanted.add(sid)
    return wanted


def _build_revision_candidates(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    merge_candidates: List[Dict[str, Any]] = []
    delete_candidates: List[Dict[str, Any]] = []
    missing_gt_candidates: List[Dict[str, Any]] = []
    quote_repair_candidates: List[Dict[str, Any]] = []

    for row in details:
        sid = str(row.get("id") or "")
        for item in row.get("items") or []:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("error_id") or "")
            label = str(item.get("forensic_label") or "")
            entry = {
                "sample_id": sid,
                "error_id": eid,
                "forensic_label": label,
                "reason": str(item.get("reason") or "")[:240],
            }
            if label in {"duplicate_root_cause", "valid_but_too_fine", "consequence_only"}:
                merge_candidates.append(entry)
            if label in {"not_error", "questionable", "duplicate_root_cause", "consequence_only"}:
                delete_candidates.append(entry)
            if label == "quote_misaligned":
                quote_repair_candidates.append(entry)
        for miss in row.get("missing_gt_candidates") or []:
            if isinstance(miss, dict):
                missing_gt_candidates.append({"sample_id": sid, **miss})

    return {
        "merge_candidates": merge_candidates,
        "delete_candidates": delete_candidates,
        "missing_gt_candidates": missing_gt_candidates,
        "quote_repair_candidates": quote_repair_candidates,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Forensic annotation audit for recall attribution.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--forensics", default="", help="common_missed_gt_report.json")
    parser.add_argument("--fp-report", default="", help="checker_fp_or_missed_label_report.json")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--priority-top-n", type=int, default=25)
    parser.add_argument("--audit-all", action="store_true", help="Audit all samples instead of priority subset.")
    parser.add_argument("--from-audit", default="", help="Map an existing annotation audit JSON to forensic labels (no LLM).")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--batch-gt-size", type=int, default=4)
    args = parser.parse_args()

    rows = _load_json(args.dataset)
    if not isinstance(rows, list):
        raise SystemExit("dataset must be a JSON array")

    if str(args.from_audit or "").strip():
        existing = _load_json(str(args.from_audit))
        if not isinstance(existing, dict):
            raise SystemExit("--from-audit must be a JSON object")
        report = _build_from_existing_audit(existing, fp_report_path=str(args.fp_report or ""))
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        for name, payload in report["revision_candidates"].items():
            (out_path.parent / f"{name}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
        return

    if args.audit_all:
        target_rows = rows
    else:
        wanted = _priority_sample_ids(
            dataset_rows=rows,
            forensics_path=str(args.forensics or ""),
            fp_report_path=str(args.fp_report or ""),
            top_n=int(args.priority_top_n),
        )
        target_rows = [r for r in rows if isinstance(r, dict) and str(r.get("id") or "") in wanted]
        if not target_rows:
            target_rows = rows[: min(25, len(rows))]

    client = _openai_client()
    details: List[Dict[str, Any]] = []
    for row in target_rows:
        sid = str(row.get("id") or "")
        gt_items = row.get("physics_error_gt") if isinstance(row.get("physics_error_gt"), list) else []
        audit = _audit_sample(
            client,
            str(args.model),
            question=str(row.get("question") or ""),
            prediction=str(row.get("prediction") or ""),
            answer=str(row.get("answer") or ""),
            gt_items=[g for g in gt_items if isinstance(g, dict)],
            timeout=float(args.timeout),
            batch_gt_size=int(args.batch_gt_size),
        )
        mapped_items = []
        label_counter: Counter = Counter()
        for item in audit.get("items") or []:
            if not isinstance(item, dict):
                continue
            forensic = _map_forensic_label(str(item.get("label") or ""), bool(item.get("quote_grounded")))
            mapped_items.append({**item, "forensic_label": forensic})
            label_counter[forensic] += 1
        details.append(
            {
                "id": sid,
                "audit_ok": bool(audit.get("ok")),
                "audit_error": audit.get("error"),
                "gt_count": len(gt_items),
                "forensic_label_counts": dict(label_counter),
                "items": mapped_items,
            }
        )

    summary_counts: Counter = Counter()
    total_items = 0
    for row in details:
        for item in row.get("items") or []:
            if isinstance(item, dict):
                summary_counts[str(item.get("forensic_label") or "unknown")] += 1
                total_items += 1

    problematic = (
        summary_counts.get("not_error", 0)
        + summary_counts.get("duplicate_root_cause", 0)
        + summary_counts.get("consequence_only", 0)
        + summary_counts.get("valid_but_too_fine", 0)
    )
    quote_misaligned = summary_counts.get("quote_misaligned", 0)

    revision = _build_revision_candidates(details)
    report = {
        "summary": {
            "model": args.model,
            "samples_audited": len(details),
            "total_gt_items": total_items,
            "forensic_label_counts": dict(summary_counts),
            "problematic_ratio": round(problematic / max(1, total_items), 4),
            "quote_misaligned_ratio": round(quote_misaligned / max(1, total_items), 4),
            "valid_root_error_ratio": round(summary_counts.get("valid_root_error", 0) / max(1, total_items), 4),
        },
        "details": details,
        "revision_candidates": revision,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for name, payload in revision.items():
        (out_path.parent / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
