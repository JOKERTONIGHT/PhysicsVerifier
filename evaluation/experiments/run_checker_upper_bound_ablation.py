#!/usr/bin/env python3
"""Analyze checker upper-bound bottlenecks and compare single vs exhaustive modes."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from evaluate_match_sensitivity import evaluate_layers  # type: ignore
from evaluate_physics_eval_sets import _load_json  # type: ignore


def _parse_checker_log(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"json_parse_failures": 0, "retry_errors": 0, "lines": 0}
    text = path.read_text(encoding="utf-8", errors="ignore")
    return {
        "json_parse_failures": len(re.findall(r"Failed to parse JSON", text)),
        "retry_errors": len(re.findall(r"Error evaluating sample", text)),
        "exhaustive_errors": len(re.findall(r"Error exhaustive sample", text)),
        "lines": len(text.splitlines()),
    }


def _diag_stats(results_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = [len(r.get("diagnostics") or []) for r in results_rows if isinstance(r, dict)]
    if not counts:
        return {"samples": 0, "avg_diagnostics": 0.0, "max_diagnostics": 0, "zero_diag_samples": 0}
    return {
        "samples": len(counts),
        "avg_diagnostics": round(sum(counts) / len(counts), 3),
        "max_diagnostics": max(counts),
        "zero_diag_samples": sum(1 for c in counts if c == 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Checker upper-bound ablation report.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--single-results", required=True)
    parser.add_argument("--exhaustive-results", default="")
    parser.add_argument("--single-log", default="")
    parser.add_argument("--exhaustive-log", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    dataset_rows = _load_json(args.dataset)
    single_rows = _load_json(args.single_results)
    if not isinstance(dataset_rows, list) or not isinstance(single_rows, list):
        raise SystemExit("dataset and single-results must be JSON arrays")

    single_layers = evaluate_layers(
        dataset_rows=dataset_rows,
        results_rows=single_rows,
        strict_iou=0.5,
        strict_cov=0.6,
        relaxed_iou=0.2,
        relaxed_cov=0.3,
    )
    exhaustive_layers = None
    exhaustive_rows: List[Dict[str, Any]] = []
    subset_ids: set = set()
    if str(args.exhaustive_results or "").strip() and Path(args.exhaustive_results).exists():
        exhaustive_rows = _load_json(str(args.exhaustive_results))
        if isinstance(exhaustive_rows, list):
            subset_ids = {str(r.get("id") or "") for r in exhaustive_rows if isinstance(r, dict)}
            subset_dataset = [r for r in dataset_rows if isinstance(r, dict) and str(r.get("id") or "") in subset_ids]
            subset_single = [r for r in single_rows if isinstance(r, dict) and str(r.get("id") or "") in subset_ids]
            exhaustive_layers = evaluate_layers(
                dataset_rows=subset_dataset,
                results_rows=exhaustive_rows,
                strict_iou=0.5,
                strict_cov=0.6,
                relaxed_iou=0.2,
                relaxed_cov=0.3,
            )
            single_subset_layers = evaluate_layers(
                dataset_rows=subset_dataset,
                results_rows=subset_single,
                strict_iou=0.5,
                strict_cov=0.6,
                relaxed_iou=0.2,
                relaxed_cov=0.3,
            )
        else:
            single_subset_layers = None
    else:
        single_subset_layers = None

    single_log = _parse_checker_log(Path(args.single_log)) if args.single_log else {}
    exhaustive_log = _parse_checker_log(Path(args.exhaustive_log)) if args.exhaustive_log else {}

    report = {
        "summary": {
            "single_strict_recall_full": single_layers.get("strict_span_recall"),
            "single_strict_recall_subset": (single_subset_layers or {}).get("strict_span_recall"),
            "single_semantic_recall_full": single_layers.get("semantic_detection_recall"),
            "single_avg_diagnostics": _diag_stats(single_rows).get("avg_diagnostics"),
            "exhaustive_subset_samples": len(subset_ids),
            "exhaustive_strict_recall_subset": (exhaustive_layers or {}).get("strict_span_recall"),
            "exhaustive_semantic_recall_subset": (exhaustive_layers or {}).get("semantic_detection_recall"),
            "exhaustive_avg_diagnostics": _diag_stats(exhaustive_rows).get("avg_diagnostics")
            if exhaustive_layers
            else None,
            "exhaustive_lift_subset": round(
                float((exhaustive_layers or {}).get("strict_span_recall") or 0.0)
                - float((single_subset_layers or {}).get("strict_span_recall") or 0.0),
                4,
            )
            if exhaustive_layers and single_subset_layers
            else None,
        },
        "single_round": {
            "layers": single_layers,
            "diagnostics": _diag_stats(single_rows),
            "log_analysis": single_log,
        },
        "exhaustive_round": {
            "layers": exhaustive_layers,
            "single_on_subset_layers": single_subset_layers,
            "diagnostics": _diag_stats(exhaustive_rows) if exhaustive_layers else None,
            "log_analysis": exhaustive_log,
            "subset_sample_ids": sorted(subset_ids),
        },
        "checker_json_failure_report": {
            "single": single_log,
            "exhaustive": exhaustive_log,
        },
        "quote_repair_success_report": {
            "note": "Exhaustive mode performs heuristic quote repair when exact quote missing.",
            "single_mode_quote_repair": False,
            "exhaustive_mode_quote_repair": True,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out.parent / "checker_json_failure_report.json").write_text(
        json.dumps(report["checker_json_failure_report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out.parent / "quote_repair_success_report.json").write_text(
        json.dumps(report["quote_repair_success_report"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
