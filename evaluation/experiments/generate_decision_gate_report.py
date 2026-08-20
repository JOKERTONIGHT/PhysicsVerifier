#!/usr/bin/env python3
"""Decision gate: choose annotation vs matching vs checker vs model-gap path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pick_recall(rows: List[Dict[str, Any]], key: str = "strict_span_recall") -> float:
    if not rows:
        return 0.0
    return float(rows[0].get(key) or 0.0)


def decide(
    *,
    forensics: Dict[str, Any],
    annotation: Dict[str, Any],
    matching: Dict[str, Any],
    checker_ablation: Dict[str, Any],
) -> Dict[str, Any]:
    ann_summary = annotation.get("summary") or {}
    problematic_ratio = float(ann_summary.get("problematic_ratio") or 0.0)
    quote_misaligned_ratio = float(ann_summary.get("quote_misaligned_ratio") or 0.0)
    missing_gt_count = len((annotation.get("revision_candidates") or {}).get("missing_gt_candidates") or [])
    missing_gt_count = max(
        missing_gt_count,
        int((ann_summary.get("missing_gt_candidates_from_fp") or 0)),
    )

    experiments = matching.get("experiments") or []
    gemini = next((e for e in experiments if "gemini" in str(e.get("label") or "").lower()), experiments[-1] if experiments else {})
    strict = float(gemini.get("strict_span_recall") or 0.0)
    paragraph = float(gemini.get("paragraph_recall") or 0.0)
    semantic = float(gemini.get("semantic_detection_recall") or 0.0)
    group = float(gemini.get("group_recall") or 0.0)

    exhaustive = checker_ablation.get("summary") or {}
    exhaustive_recall = float(exhaustive.get("exhaustive_strict_recall_subset") or exhaustive.get("exhaustive_strict_recall") or 0.0)
    single_recall = float(exhaustive.get("single_strict_recall_subset") or exhaustive.get("single_strict_recall") or 0.0)
    exhaustive_lift = float(exhaustive.get("exhaustive_lift_subset") or exhaustive.get("exhaustive_lift") or 0.0)

    paths: List[str] = []
    rationale: List[str] = []

    if problematic_ratio >= 0.15 or missing_gt_count >= 8:
        paths.append("A_annotation")
        rationale.append(
            f"problematic_ratio={problematic_ratio:.3f} (threshold 0.15) or missing_gt={missing_gt_count}"
        )
    if quote_misaligned_ratio >= 0.10 or (paragraph - strict) >= 0.08 or (semantic - strict) >= 0.15:
        paths.append("B_matching")
        rationale.append(
            f"quote_misaligned_ratio={quote_misaligned_ratio:.3f}; "
            f"paragraph-strict gap={(paragraph - strict):.3f}; "
            f"semantic-strict gap={(semantic - strict):.3f}"
        )
    if exhaustive_lift >= 0.05:
        paths.append("C_checker")
        rationale.append(f"exhaustive checker lift over single-round={exhaustive_lift:.3f}")
    if (
        problematic_ratio < 0.10
        and (paragraph - strict) < 0.05
        and exhaustive_lift < 0.05
        and strict < 0.35
        and semantic < 0.45
    ):
        paths.append("D_model_gap")
        rationale.append("annotation/matching/checker optimizations insufficient; strict+semantic recall remain low")

    if not paths:
        paths = ["B_matching"]
        rationale.append("default to matching optimization when no strong signal")

    primary = paths[0]
    if "A_annotation" in paths and "B_matching" in paths:
        primary = "A+B_mixed"
    elif len(paths) > 1:
        primary = "+".join(paths)

    conditional_relabel = "A_annotation" in paths or "A+B_mixed" == primary or primary.startswith("A")
    actions: List[str] = []
    if conditional_relabel:
        actions.extend(["merge_duplicate_root_cause", "delete_not_error_consequence", "add_missing_gt", "repair_quotes"])
    if "B_matching" in paths:
        actions.extend(["paragraph_fallback_metric", "group_aware_metric", "quote_repair_pipeline"])
    if "C_checker" in paths:
        actions.extend(["adopt_exhaustive_checker_as_upper_bound", "json_repair_hardening"])
    if "D_model_gap" in paths:
        actions.extend(["multi_model_adjudicated_upper_bound", "report_method_gap"])

    return {
        "primary_path": primary,
        "candidate_paths": paths,
        "conditional_relabel_recommended": conditional_relabel,
        "recommended_actions": actions,
        "evidence": {
            "problematic_ratio": problematic_ratio,
            "quote_misaligned_ratio": quote_misaligned_ratio,
            "missing_gt_candidates": missing_gt_count,
            "gemini_strict_recall": strict,
            "gemini_paragraph_recall": paragraph,
            "gemini_semantic_recall": semantic,
            "gemini_group_recall": group,
            "exhaustive_lift": exhaustive_lift,
            "common_missed_gt_count": (forensics.get("common_missed_gt") or {}).get("common_count"),
        },
        "rationale": rationale,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate recall attribution decision gate report.")
    parser.add_argument("--forensics", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--matching", required=True)
    parser.add_argument("--checker-ablation", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    forensics = _load(args.forensics)
    annotation = _load(args.annotation)
    matching = _load(args.matching)
    checker_ablation = _load(args.checker_ablation)

    decision = decide(
        forensics=forensics,
        annotation=annotation,
        matching=matching,
        checker_ablation=checker_ablation,
    )

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        "# Dataset Revision Decision Gate",
        "",
        f"**Primary path:** `{decision['primary_path']}`",
        f"**Conditional relabel recommended:** {decision['conditional_relabel_recommended']}",
        "",
        "## Evidence",
    ]
    for k, v in decision["evidence"].items():
        md_lines.append(f"- {k}: {v}")
    md_lines.extend(["", "## Rationale"])
    for r in decision["rationale"]:
        md_lines.append(f"- {r}")
    md_lines.extend(["", "## Recommended Actions"])
    for a in decision["recommended_actions"]:
        md_lines.append(f"- {a}")

    Path(args.output_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(json.dumps(decision, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
