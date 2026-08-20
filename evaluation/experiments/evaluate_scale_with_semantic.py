#!/usr/bin/env python3
"""Run location + semantic dual evaluation and near-miss diagnostics for scale A/B runs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _near_miss_ratio(failure_analysis: Dict[str, Any]) -> Dict[str, Any]:
    summary = failure_analysis.get("summary") if isinstance(failure_analysis, dict) else {}
    zero_gt = summary.get("zero_match_gt_causes") if isinstance(summary.get("zero_match_gt_causes"), dict) else {}
    fp_items = summary.get("fp_item_causes") if isinstance(summary.get("fp_item_causes"), dict) else {}

    fn_near = int(zero_gt.get("semantic_near_miss") or 0)
    fn_gap = int(zero_gt.get("semantic_gap") or 0)
    fn_loc = int(zero_gt.get("location_failure") or 0)
    fn_total = max(1, fn_near + fn_gap + fn_loc)

    fp_near = int(fp_items.get("semantic_near_miss") or 0)
    fp_broad = int(fp_items.get("rule_too_broad") or 0)
    fp_irr = int(fp_items.get("irrelevant_trigger") or 0)
    fp_total = max(1, fp_near + fp_broad + fp_irr)

    return {
        "fn_semantic_near_miss_ratio": round(fn_near / fn_total, 4),
        "fn_semantic_gap_ratio": round(fn_gap / fn_total, 4),
        "fn_location_failure_ratio": round(fn_loc / fn_total, 4),
        "fp_semantic_near_miss_ratio": round(fp_near / fp_total, 4),
        "fp_rule_too_broad_ratio": round(fp_broad / fp_total, 4),
        "fp_irrelevant_trigger_ratio": round(fp_irr / fp_total, 4),
    }


def _quote_exact_ratio(metrics: Dict[str, Any], verifier_results: List[Dict[str, Any]], dataset_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ds_by_id = {str(r.get("id") or ""): r for r in dataset_rows if isinstance(r, dict)}
    total_quotes = 0
    exact_quotes = 0
    for row in verifier_results:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "")
        answer = str(ds_by_id.get(sid, {}).get("prediction") or "")
        for d in row.get("diagnostics") or []:
            if not isinstance(d, dict):
                continue
            ev = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
            quote = str(ev.get("quote") or "").strip()
            if not quote:
                continue
            total_quotes += 1
            if quote in answer:
                exact_quotes += 1
    ratio = round(exact_quotes / max(1, total_quotes), 4)
    return {
        "quote_exact_substring_count": exact_quotes,
        "quote_total_count": total_quotes,
        "quote_exact_substring_ratio": ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dual location/semantic evaluation for scale A/B.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="run")
    parser.add_argument("--semantic-match-model", default="deepseek-v4-pro")
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--skip-failure-analysis", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    location_metrics_path = out_dir / "error_metrics.json"
    semantic_metrics_path = out_dir / "semantic_metrics.json"
    failure_path = out_dir / "failure_analysis.json"
    by_rule_path = out_dir / "failure_analysis_by_rule.json"
    combined_path = out_dir / "dual_eval_summary.json"

    _run(
        [
            sys.executable,
            "scripts/evaluate_physics_eval_sets.py",
            "--dataset",
            args.dataset,
            "--results",
            args.results,
            "--audit",
            args.audit,
            "--output",
            str(location_metrics_path),
            "--match-mode",
            "location",
        ]
    )

    if not args.skip_semantic:
        _run(
            [
                sys.executable,
                "scripts/evaluate_physics_eval_sets.py",
                "--dataset",
                args.dataset,
                "--results",
                args.results,
                "--audit",
                args.audit,
                "--output",
                str(semantic_metrics_path),
                "--match-mode",
                "semantic",
                "--semantic-match-model",
                args.semantic_match_model,
            ]
        )

    if not args.skip_failure_analysis:
        _run(
            [
                sys.executable,
                "scripts/analyze_scale_failure_cases.py",
                "--metrics",
                str(location_metrics_path),
                "--verifier-results",
                args.results,
                "--dataset",
                args.dataset,
                "--output",
                str(failure_path),
                "--by-rule-output",
                str(by_rule_path),
            ]
        )

    location_metrics = _load_json(location_metrics_path)
    semantic_metrics = _load_json(semantic_metrics_path) if semantic_metrics_path.exists() else {}
    failure_analysis = _load_json(failure_path) if failure_path.exists() else {}
    verifier_rows = json.loads(Path(args.results).read_text(encoding="utf-8"))
    dataset_rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))

    loc_summary = location_metrics.get("summary") if isinstance(location_metrics, dict) else {}
    sem_summary = semantic_metrics.get("summary") if isinstance(semantic_metrics, dict) else {}

    combined = {
        "label": args.label,
        "location": loc_summary,
        "semantic": sem_summary,
        "near_miss": _near_miss_ratio(failure_analysis),
        "quote_quality": _quote_exact_ratio(location_metrics, verifier_rows, dataset_rows),
        "paths": {
            "location_metrics": str(location_metrics_path),
            "semantic_metrics": str(semantic_metrics_path) if semantic_metrics_path.exists() else None,
            "failure_analysis": str(failure_path) if failure_path.exists() else None,
            "failure_analysis_by_rule": str(by_rule_path) if by_rule_path.exists() else None,
        },
    }
    if isinstance(failure_analysis.get("by_rule"), dict):
        combined["top_fp_rules"] = failure_analysis["by_rule"].get("top_fp_rules", [])[:10]
        combined["top_missed_gt_themes"] = failure_analysis["by_rule"].get("top_missed_gt_themes", [])[:10]

    combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(combined, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
