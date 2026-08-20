#!/usr/bin/env python3
"""Replay aggregator/validator post-processing on saved verifier results."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.diagnostic_aggregator import DiagnosticAggregator
from core.diagnostic_validator import DiagnosticValidator


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rule_record_from_diagnostic(d: Dict[str, Any]) -> Dict[str, Any]:
    rule_match = d.get("rule_match") if isinstance(d.get("rule_match"), dict) else {}
    publish_gate = rule_match.get("publish_gate") if isinstance(rule_match.get("publish_gate"), dict) else {}
    return {
        "rule_id": str(d.get("rule") or ""),
        "rule": {
            "id": str(d.get("rule") or ""),
            "precision": {
                "negative_conditions": publish_gate.get("negative_condition_hits") or [],
                "evidence_requirements": publish_gate.get("evidence_requirement_hits") or [],
            },
        },
        "score": float(rule_match.get("score") or d.get("release_gate", {}).get("rule_score") or 0.0),
        "publish_gate": publish_gate,
    }


def replay(
    *,
    dataset_rows: List[Dict[str, Any]],
    verifier_rows: List[Dict[str, Any]],
    enable_validator: bool,
    enable_aggregator: bool,
) -> List[Dict[str, Any]]:
    ds_by_id = {str(r.get("id") or ""): r for r in dataset_rows if isinstance(r, dict)}
    validator = DiagnosticValidator(use_llm=False)
    aggregator = DiagnosticAggregator()
    out_rows: List[Dict[str, Any]] = []

    for row in verifier_rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "")
        sample = ds_by_id.get(sid, {})
        question = str(sample.get("question") or "")
        reference_answer = str(sample.get("reference_answer") or sample.get("answer") or "")
        student_solution = str(sample.get("prediction") or "")
        diagnostics = [deepcopy(d) for d in (row.get("diagnostics") or []) if isinstance(d, dict)]
        rule_records = [_rule_record_from_diagnostic(d) for d in diagnostics]

        if enable_aggregator and diagnostics:
            diagnostics, _ = aggregator.aggregate(diagnostics, rule_records=rule_records)
            rule_records = [_rule_record_from_diagnostic(d) for d in diagnostics]

        if enable_validator and diagnostics:
            diagnostics, _ = validator.validate_batch(
                question=question,
                reference_answer=reference_answer,
                student_solution=student_solution,
                diagnostics=diagnostics,
                rule_records=rule_records,
            )

        new_row = {
            "id": row.get("id"),
            "topic": row.get("topic"),
            "verifier": row.get("verifier"),
            "diagnostics": diagnostics,
            "score": -1.0 * len(diagnostics),
        }
        out_rows.append(new_row)
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay post-process pipeline on saved verifier outputs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--enable-validator", action="store_true")
    parser.add_argument("--enable-aggregator", action="store_true")
    parser.add_argument("--skip-semantic", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_rows = _load(args.dataset)
    verifier_rows = _load(args.results)
    if not isinstance(dataset_rows, list) or not isinstance(verifier_rows, list):
        raise SystemExit("dataset/results must be JSON arrays")

    replayed = replay(
        dataset_rows=dataset_rows,
        verifier_rows=verifier_rows,
        enable_validator=bool(args.enable_validator),
        enable_aggregator=bool(args.enable_aggregator),
    )

    pred_path = out_dir / "error_verifier_results.json"
    pred_path.write_text(json.dumps(replayed, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_src = Path(args.audit)
    if audit_src.exists():
        (out_dir / "error_symbolic_audit.json").write_text(audit_src.read_text(encoding="utf-8"), encoding="utf-8")

    dataset_copy = out_dir / "error_eval_dataset_100.json"
    if not dataset_copy.exists():
        dataset_copy.write_text(Path(args.dataset).read_text(encoding="utf-8"), encoding="utf-8")

    eval_cmd = [
        sys.executable,
        "scripts/evaluate_scale_with_semantic.py",
        "--dataset",
        str(dataset_copy),
        "--results",
        str(pred_path),
        "--audit",
        str(out_dir / "error_symbolic_audit.json"),
        "--output-dir",
        str(out_dir),
        "--label",
        args.label,
    ]
    if args.skip_semantic:
        eval_cmd.append("--skip-semantic")
    subprocess.run(eval_cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()
