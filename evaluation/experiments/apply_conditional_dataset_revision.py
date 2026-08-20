#!/usr/bin/env python3
"""Apply conditional dataset revision (merge/delete/repair) from forensic audit."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


DELETE_LABELS = frozenset({"not_error", "duplicate_root_cause", "consequence_only", "questionable"})
MERGE_LABELS = frozenset({"duplicate_root_cause", "valid_but_too_fine", "consequence_only"})


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tokenize(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(t) >= 3}


def _collect_forensic_labels(audit: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for row in audit.get("details") or []:
        if not isinstance(row, dict):
            continue
        for item in row.get("items") or []:
            if not isinstance(item, dict):
                continue
            eid = str(item.get("error_id") or "")
            label = str(item.get("forensic_label") or item.get("label") or "")
            if eid:
                out[eid] = label
    return out


def _repair_quote(gt: Dict[str, Any], prediction: str) -> Tuple[Dict[str, Any], bool]:
    quote = str(gt.get("answer_quote") or "").strip()
    if quote and quote in prediction:
        return gt, False
    err = str(gt.get("error_text") or "")
    tokens = [t for t in re.findall(r"[A-Za-z0-9=+\-*/^()\\[\]{}.,]+", err) if len(t) >= 4]
    best = ""
    for tok in sorted(tokens, key=len, reverse=True)[:12]:
        if tok in prediction and len(tok) > len(best):
            best = tok
    if best:
        patched = dict(gt)
        patched["answer_quote"] = best
        return patched, True
    return gt, False


def apply_revision(
    *,
    dataset_rows: List[Dict[str, Any]],
    audit: Dict[str, Any],
    decision: Dict[str, Any],
    force: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not force and not bool(decision.get("conditional_relabel_recommended")):
        return dataset_rows, {"skipped": True, "reason": "decision gate did not recommend relabel"}

    labels = _collect_forensic_labels(audit)
    delete_ids: Set[str] = {eid for eid, lab in labels.items() if lab in DELETE_LABELS}
    merge_ids: Set[str] = {eid for eid, lab in labels.items() if lab in MERGE_LABELS}

    removed = 0
    merged = 0
    repaired = 0
    before_gt = 0
    after_gt = 0
    actions: List[Dict[str, Any]] = []
    revised_rows: List[Dict[str, Any]] = []

    for row in dataset_rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "")
        prediction = str(row.get("prediction") or "")
        gt_items = [g for g in (row.get("physics_error_gt") or []) if isinstance(g, dict)]
        before_gt += len(gt_items)

        kept: List[Dict[str, Any]] = []
        for g in gt_items:
            eid = str(g.get("error_id") or "")
            if eid in delete_ids:
                removed += 1
                actions.append({"action": "delete", "sample_id": sid, "error_id": eid, "label": labels.get(eid)})
                continue
            patched, did_repair = _repair_quote(g, prediction)
            if did_repair:
                repaired += 1
                actions.append({"action": "repair_quote", "sample_id": sid, "error_id": eid})
            kept.append(patched)

        # merge: within sample, collapse merge_ids sharing paragraph + token overlap
        by_para: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for g in kept:
            by_para[int(g.get("paragraph_index") or 0)].append(g)
        final_kept: List[Dict[str, Any]] = []
        for para, group in by_para.items():
            if len(group) <= 1:
                final_kept.extend(group)
                continue
            rep = group[0]
            absorbed: List[str] = []
            for g in group[1:]:
                eid = str(g.get("error_id") or "")
                if eid in merge_ids and _tokenize(str(rep.get("error_text") or "")) & _tokenize(str(g.get("error_text") or "")):
                    merged += 1
                    absorbed.append(eid)
                    actions.append({"action": "merge", "sample_id": sid, "keep": rep.get("error_id"), "drop": eid})
                else:
                    final_kept.append(g)
            final_kept.append(rep)
        after_gt += len(final_kept)

        out_row = dict(row)
        out_row["physics_error_gt"] = final_kept
        out_row["physics_error_gt_valid_count"] = len(final_kept)
        out_row["physics_error_examples"] = [{"error": str(g.get("error_text") or "")} for g in final_kept]
        revised_rows.append(out_row)

    report = {
        "skipped": False,
        "before_gt_count": before_gt,
        "after_gt_count": after_gt,
        "removed_count": removed,
        "merged_count": merged,
        "repaired_quote_count": repaired,
        "actions": actions[:500],
    }
    return revised_rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Conditional dataset revision from forensic audit.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--annotation-forensics", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--force", action="store_true", help="Apply even if decision gate says no relabel.")
    args = parser.parse_args()

    rows = _load(args.dataset)
    audit = _load(args.annotation_forensics)
    decision = _load(args.decision)
    if not isinstance(rows, list):
        raise SystemExit("dataset must be a JSON array")

    revised, report = apply_revision(
        dataset_rows=rows,
        audit=audit,
        decision=decision,
        force=bool(args.force),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not report.get("skipped"):
        out_path.write_text(json.dumps(revised, ensure_ascii=False, indent=2), encoding="utf-8")
    report["output"] = str(out_path)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
