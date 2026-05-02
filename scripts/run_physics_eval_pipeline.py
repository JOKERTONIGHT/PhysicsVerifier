from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path
from typing import List


def _run(cmd: str) -> None:
    print(f"[RUN] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise SystemExit(f"Command failed ({res.returncode}): {cmd}")


def _resolve_existing_dataset(outdir: Path, preferred: Path, pattern: str, label: str) -> Path:
    if preferred.exists():
        return preferred
    candidates: List[Path] = sorted(outdir.glob(pattern))
    if len(candidates) == 1:
        print(f"[INFO] {label} dataset auto-resolved to {candidates[0]}")
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(str(x.name) for x in candidates[:8])
        raise SystemExit(
            f"{label} dataset not found at expected path {preferred}. Multiple candidates found: {names}. "
            f"Please specify matching recall/precision sizes or run without --skip-build."
        )
    raise SystemExit(
        f"{label} dataset not found at expected path {preferred}, and no files match pattern {pattern} in {outdir}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end independent error-level and question-level evaluation pipeline.")
    parser.add_argument("--python", type=str, default="./.venv/bin/python")
    parser.add_argument("--input", type=str, default="data/physics_rubric_data_1000.json")
    parser.add_argument("--recall-input", type=str, default="data/evaluation_sample_1000_expansion.json")
    parser.add_argument("--precision-input", type=str, default="data/physics_rubric_data_1000.json")
    parser.add_argument("--recall-size", type=int, default=20)
    parser.add_argument("--precision-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--strong-model", type=str, default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--check-model", type=str, default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--max-errors", type=int, default=0, help="0 means exhaustive GT extraction mode.")
    parser.add_argument("--max-recall-scan", type=int, default=500)
    parser.add_argument("--min-valid-gt-per-sample", type=int, default=1)
    parser.add_argument("--disable-agentic", action="store_true", help="Disable agentic post-check in verifier for stability.")
    parser.add_argument("--run-quality-audit", action="store_true")
    parser.add_argument("--require-quality-pass", action="store_true")
    parser.add_argument("--min-locatable-ratio", type=float, default=0.70)
    parser.add_argument("--min-avg-errors", type=float, default=2.0)
    parser.add_argument("--max-generic-ratio", type=float, default=0.25)
    parser.add_argument("--output-dir", type=str, default="results/eval_pipeline")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--skip-error-eval", action="store_true")
    parser.add_argument("--skip-question-eval", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    error_dataset = outdir / f"error_eval_dataset_{args.recall_size}.json"
    question_dataset = outdir / f"question_eval_dataset_{args.recall_size}_{args.precision_size}.json"
    question_right_only_dataset = outdir / f"question_right_only_{args.precision_size}.json"

    if args.skip_build:
        error_dataset = _resolve_existing_dataset(outdir, error_dataset, "error_eval_dataset_*.json", "Error-level")
        question_dataset = _resolve_existing_dataset(outdir, question_dataset, "question_eval_dataset_*.json", "Question-level")

    error_results = outdir / "error_top_down_results.json"
    error_audit = outdir / "error_symbolic_audit.json"
    question_results = outdir / "question_top_down_results.json"
    question_audit = outdir / "question_symbolic_audit.json"
    error_metrics_output = outdir / "error_metrics.json"
    question_metrics_output = outdir / "question_metrics.json"
    quality_output = outdir / "error_quality_audit.json"

    py = shlex.quote(args.python)

    if not args.skip_build:
        _run(
            f"{py} scripts/build_physics_eval_sets.py "
            f"--input {shlex.quote(args.input)} "
            f"--recall-input {shlex.quote(args.recall_input)} "
            f"--precision-input {shlex.quote(args.precision_input)} "
            f"--error-output {shlex.quote(str(error_dataset))} "
            f"--question-output {shlex.quote(str(question_dataset))} "
            f"--precision-output {shlex.quote(str(question_right_only_dataset))} "
            f"--recall-size {args.recall_size} "
            f"--precision-size {args.precision_size} "
            f"--seed {args.seed} "
            f"--strong-model {shlex.quote(args.strong_model)} "
            f"--max-errors {args.max_errors} "
            f"--max-recall-scan {args.max_recall_scan} "
            f"--min-valid-gt-per-sample {args.min_valid_gt_per_sample}"
        )

    if args.run_quality_audit:
        _run(
            f"{py} scripts/audit_eval_set_quality.py "
            f"--recall-dataset {shlex.quote(str(error_dataset))} "
            f"--output {shlex.quote(str(quality_output))} "
            f"--min-locatable-ratio {args.min_locatable_ratio} "
            f"--min-avg-errors {args.min_avg_errors} "
            f"--max-generic-ratio {args.max_generic_ratio}"
        )
        if args.require_quality_pass:
            import json

            quality = json.loads(quality_output.read_text(encoding="utf-8"))
            if not bool(quality.get("quality_gate_passed")):
                raise SystemExit(f"Quality gate failed: {quality.get('quality_gate_issues')}")

    if not args.skip_run:
        _run(
            f"{py} scripts/run_top_down.py "
            f"--input {shlex.quote(str(error_dataset))} "
            f"--output {shlex.quote(str(error_results))} "
            f"--symbolic-output {shlex.quote(str(error_audit))} "
            f"--model {shlex.quote(args.check_model)} "
            + ("--no-agentic " if args.disable_agentic else "") +
            f"--unified-catalog catalogs/rules_unified.json"
        )
        _run(
            f"{py} scripts/run_top_down.py "
            f"--input {shlex.quote(str(question_dataset))} "
            f"--output {shlex.quote(str(question_results))} "
            f"--symbolic-output {shlex.quote(str(question_audit))} "
            f"--model {shlex.quote(args.check_model)} "
            + ("--no-agentic " if args.disable_agentic else "") +
            f"--unified-catalog catalogs/rules_unified.json"
        )
    if not args.skip_error_eval:
        _run(
            f"{py} scripts/evaluate_physics_eval_sets.py "
            f"--dataset {shlex.quote(str(error_dataset))} "
            f"--results {shlex.quote(str(error_results))} "
            f"--audit {shlex.quote(str(error_audit))} "
            f"--output {shlex.quote(str(error_metrics_output))} "
            f"--match-mode location"
        )

    if not args.skip_question_eval:
        _run(
            f"{py} scripts/evaluate_question_level_sets.py "
            f"--dataset {shlex.quote(str(question_dataset))} "
            f"--results {shlex.quote(str(question_results))} "
            f"--audit {shlex.quote(str(question_audit))} "
            f"--output {shlex.quote(str(question_metrics_output))}"
        )

    print("Done.")
    print(f"Error dataset: {error_dataset}")
    print(f"Error checker output: {error_results}")
    print(f"Error metrics: {error_metrics_output}")
    print(f"Question dataset: {question_dataset}")
    print(f"Question checker output: {question_results}")
    print(f"Question metrics: {question_metrics_output}")
    if args.run_quality_audit:
        print(f"Quality audit: {quality_output}")


if __name__ == "__main__":
    main()
