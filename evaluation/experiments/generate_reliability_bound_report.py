#!/usr/bin/env python3
"""Final reliability, upper-bound matrix, and method gap report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(path: str) -> Any:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _metric_from_error_metrics(path: str) -> Dict[str, Any]:
    data = _load(path)
    summary = data.get("summary") if isinstance(data, dict) else {}
    return summary if isinstance(summary, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reliability and upper-bound matrix report.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--forensics", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--matching", required=True)
    parser.add_argument("--checker-ablation", required=True)
    parser.add_argument("--decision", required=True)
    parser.add_argument("--revision-report", default="")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matching = _load(args.matching)
    decision = _load(args.decision)
    forensics = _load(args.forensics)
    annotation = _load(args.annotation)
    checker = _load(args.checker_ablation)
    revision = _load(args.revision_report) if args.revision_report else {}

    experiments = matching.get("experiments") or []
    upper_bound_matrix = {
        "dataset": args.dataset,
        "layers_by_experiment": experiments,
        "rule_baseline_note": "See scale_1500_cleaned error_metrics for rule recall baseline (~11.6%).",
        "checker_ablation_summary": checker.get("summary") or {},
    }

    method_gap = {
        "common_missed_gt_count": (forensics.get("common_missed_gt") or {}).get("common_count"),
        "likely_missing_gt_fp_count": forensics.get("likely_missing_gt_fp_count"),
        "annotation_problematic_ratio": (annotation.get("summary") or {}).get("problematic_ratio"),
        "decision_primary_path": decision.get("primary_path"),
        "conditional_relabel_recommended": decision.get("conditional_relabel_recommended"),
        "revision_applied": not bool(revision.get("skipped")),
        "revision_delta_gt": (revision.get("before_gt_count", 0) - revision.get("after_gt_count", 0))
        if revision
        else 0,
        "interpretation": [],
    }

    exps = {str(e.get("label")): e for e in experiments}
    if "gemini_flash" in exps and "rules_baseline" in exps:
        method_gap["interpretation"].append(
            "Gemini strict recall minus rules baseline indicates non-rule checker headroom."
        )
    if exps.get("gemini_flash", {}).get("semantic_detection_recall", 0) - exps.get("gemini_flash", {}).get(
        "strict_span_recall", 0
    ) >= 0.08:
        method_gap["interpretation"].append("Large semantic-vs-strict gap suggests localization/matching bottleneck.")

    recall_cause = {
        "aggregate_unmatched_causes": forensics.get("aggregate_unmatched_causes"),
        "experiments": forensics.get("experiments"),
        "annotation_forensics_summary": annotation.get("summary"),
        "matching_sensitivity_summary": {
            e.get("label"): {
                "strict_span_recall": e.get("strict_span_recall"),
                "paragraph_recall": e.get("paragraph_recall"),
                "semantic_detection_recall": e.get("semantic_detection_recall"),
                "group_recall": e.get("group_recall"),
            }
            for e in experiments
        },
        "decision": decision,
    }

    (out_dir / "recall_cause_diagnosis_report.json").write_text(
        json.dumps(recall_cause, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "annotation_forensics_report.json").write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "matching_sensitivity_report.json").write_text(
        json.dumps(matching, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "upper_bound_checker_ablation.json").write_text(
        json.dumps(checker, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "upper_bound_matrix.json").write_text(
        json.dumps(upper_bound_matrix, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "method_gap_report.json").write_text(
        json.dumps(method_gap, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [
        "# Recall Attribution Final Report",
        "",
        f"- Dataset: `{args.dataset}`",
        f"- Decision path: `{decision.get('primary_path')}`",
        f"- Conditional relabel: `{decision.get('conditional_relabel_recommended')}`",
        "",
        "## Annotation reliability",
        f"- Problematic ratio: {(annotation.get('summary') or {}).get('problematic_ratio')}",
        f"- Quote misaligned ratio: {(annotation.get('summary') or {}).get('quote_misaligned_ratio')}",
        "",
        "## Matching sensitivity (Gemini if present)",
    ]
    gem = next((e for e in experiments if "gemini" in str(e.get("label")).lower()), experiments[0] if experiments else {})
    for k in ["strict_span_recall", "paragraph_recall", "semantic_detection_recall", "group_recall"]:
        md.append(f"- {k}: {gem.get(k)}")
    md.extend(["", "## Checker ablation", json.dumps(checker.get("summary") or {}, ensure_ascii=False, indent=2)])
    (out_dir / "reliability_and_bound_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "method_gap": method_gap}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
