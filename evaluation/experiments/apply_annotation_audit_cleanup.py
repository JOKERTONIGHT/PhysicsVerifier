#!/usr/bin/env python3
"""Filter eval GT annotations using reasonableness audit labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


REMOVE_LABELS = frozenset({"not_error", "duplicate", "questionable", "over_granular"})


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_removals(audit: Dict[str, Any]) -> Tuple[Set[str], Dict[str, int], List[Dict[str, Any]]]:
    remove_ids: Set[str] = set()
    label_counts: Dict[str, int] = {}
    removal_log: List[Dict[str, Any]] = []
    for row in audit.get("details") or []:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "")
        for item in row.get("items") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "").strip().lower()
            label_counts[label] = label_counts.get(label, 0) + 1
            eid = str(item.get("error_id") or "").strip()
            if label in REMOVE_LABELS and eid:
                remove_ids.add(eid)
                removal_log.append(
                    {
                        "sample_id": sid,
                        "error_id": eid,
                        "label": label,
                        "reason": str(item.get("reason") or "")[:300],
                    }
                )
    return remove_ids, label_counts, removal_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove questionable GT items based on annotation audit labels.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--audit", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--report", type=str, default="")
    parser.add_argument("--backup", type=str, default="")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    audit_path = Path(args.audit)
    output_path = Path(args.output)
    rows = _load_json(dataset_path)
    if not isinstance(rows, list):
        raise SystemExit("Dataset must be a JSON array.")

    audit = _load_json(audit_path)
    if not isinstance(audit, dict):
        raise SystemExit("Audit must be a JSON object.")

    remove_ids, label_counts, removal_log = _collect_removals(audit)
    before = 0
    after = 0
    cleaned: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        gt_items = row.get("physics_error_gt") if isinstance(row.get("physics_error_gt"), list) else []
        before += len(gt_items)
        kept = [g for g in gt_items if isinstance(g, dict) and str(g.get("error_id") or "") not in remove_ids]
        after += len(kept)
        out_row = dict(row)
        out_row["physics_error_gt"] = kept
        out_row["physics_error_gt_valid_count"] = len(kept)
        out_row["physics_error_examples"] = [{"error": str(g.get("error_text") or "")} for g in kept if str(g.get("error_text") or "").strip()]
        cleaned.append(out_row)

    if args.backup:
        backup_path = Path(args.backup)
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        backup_path.write_text(dataset_path.read_text(encoding="utf-8"), encoding="utf-8")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {
        "dataset": str(dataset_path),
        "audit": str(audit_path),
        "output": str(output_path),
        "label_counts": label_counts,
        "removed_labels": sorted(REMOVE_LABELS),
        "removed_error_ids": len(remove_ids),
        "gt_before": before,
        "gt_after": after,
        "samples": len(cleaned),
        "removal_log": removal_log,
    }
    report_path = Path(args.report) if str(args.report or "").strip() else output_path.with_suffix(".cleanup_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    print(json.dumps({k: report[k] for k in ("gt_before", "gt_after", "removed_error_ids", "samples", "output", "report")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
