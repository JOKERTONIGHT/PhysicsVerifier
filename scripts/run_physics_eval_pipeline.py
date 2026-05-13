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
    parser.add_argument("--unified-catalog", type=str, default="", help="Path to unified rule catalog JSON for run_verifier.py.")
    parser.add_argument(
        "--no-symbolic-check",
        action="store_true",
        help="Disable the experience-code symbolic verification (on by default).",
    )
    parser.add_argument(
        "--experience-code-manifest",
        type=str,
        default="results/experience_symbolic_program_manifest_v2_unified.json",
        help="Forwarded to run_verifier.py.",
    )
    parser.add_argument(
        "--experience-code-module",
        type=str,
        default="symbolic.generated_experience_checks_v2_unified",
        help="Forwarded to run_verifier.py.",
    )
    parser.add_argument(
        "--symbolic-topic-check-limit",
        type=int,
        default=40,
        help="Forwarded to run_verifier.py.",
    )
    parser.add_argument(
        "--unified-rule-top-n",
        type=int,
        default=None,
        help="Forwarded to run_verifier.py (unified v2 rule pool width per sample).",
    )
    parser.add_argument(
        "--min-diagnostic-rule-score",
        type=float,
        default=None,
        help="Forwarded to run_verifier.py.",
    )
    # Legacy flag, kept for backward compatibility (no-op).
    parser.add_argument("--disable-agentic", action="store_true", help="(deprecated, no-op)")
    parser.add_argument(
        "--max-per-sample",
        type=int,
        default=12,
        help="Forwarded to run_verifier.py: cap published diagnostics per sample (<=0 disables). Default 12 improves precision on long rollouts.",
    )
    parser.add_argument(
        "--max-per-paragraph",
        type=int,
        default=2,
        help="Forwarded to run_verifier.py: cap diagnostics per paragraph (<=0 disables). Default 2 reduces redundant alarms.",
    )
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
    parser.add_argument(
        "--verifier-progress-interval",
        type=int,
        default=10,
        metavar="N",
        help="Forwarded to run_verifier.py --progress-interval for each verifier pass (0 disables).",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    error_dataset = outdir / f"error_eval_dataset_{args.recall_size}.json"
    question_dataset = outdir / f"question_eval_dataset_{args.recall_size}_{args.precision_size}.json"
    question_right_only_dataset = outdir / f"question_right_only_{args.precision_size}.json"

    if args.skip_build:
        error_dataset = _resolve_existing_dataset(outdir, error_dataset, "error_eval_dataset_*.json", "Error-level")
        question_dataset = _resolve_existing_dataset(outdir, question_dataset, "question_eval_dataset_*.json", "Question-level")

    error_results = outdir / "error_verifier_results.json"
    error_audit = outdir / "error_symbolic_audit.json"
    question_results = outdir / "question_verifier_results.json"
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

    cap_parts: List[str] = []
    if int(args.max_per_sample) > 0:
        cap_parts.append(f"--max-per-sample {int(args.max_per_sample)}")
    else:
        cap_parts.append("--max-per-sample 0")
    if int(args.max_per_paragraph) > 0:
        cap_parts.append(f"--max-per-paragraph {int(args.max_per_paragraph)}")
    else:
        cap_parts.append("--max-per-paragraph 0")
    cap_flag = " ".join(cap_parts) + " "

    vf_extra = ""
    if args.unified_rule_top_n is not None:
        vf_extra += f" --unified-rule-top-n {int(args.unified_rule_top_n)}"
    if args.min_diagnostic_rule_score is not None:
        vf_extra += f" --min-diagnostic-rule-score {float(args.min_diagnostic_rule_score)}"
    vf_extra += f" --progress-interval {max(0, int(args.verifier_progress_interval))}"

    if not args.skip_run:
        catalog_flag = f"--unified-catalog {shlex.quote(args.unified_catalog)}" if args.unified_catalog else ""
        symbolic_flag_parts: List[str] = [
            f"--experience-code-manifest {shlex.quote(args.experience_code_manifest)}",
            f"--experience-code-module {shlex.quote(args.experience_code_module)}",
            f"--symbolic-topic-check-limit {int(args.symbolic_topic_check_limit)}",
        ]
        if args.no_symbolic_check:
            symbolic_flag_parts.append("--no-symbolic-check")
        symbolic_flag = " ".join(symbolic_flag_parts) + " "
        _run(
            f"{py} scripts/run_verifier.py "
            f"--input {shlex.quote(str(error_dataset))} "
            f"--output {shlex.quote(str(error_results))} "
            f"--symbolic-output {shlex.quote(str(error_audit))} "
            f"--model {shlex.quote(args.check_model)} "
            + cap_flag
            + symbolic_flag
            + (catalog_flag + " " if catalog_flag else "")
            + vf_extra
        )
        _run(
            f"{py} scripts/run_verifier.py "
            f"--input {shlex.quote(str(question_dataset))} "
            f"--output {shlex.quote(str(question_results))} "
            f"--symbolic-output {shlex.quote(str(question_audit))} "
            f"--model {shlex.quote(args.check_model)} "
            + cap_flag
            + symbolic_flag
            + (catalog_flag + " " if catalog_flag else "")
            + vf_extra
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
    print(f"  Error dataset:          {error_dataset}")
    print(f"  Error verifier output:  {error_results}")
    print(f"  Error metrics:          {error_metrics_output}")
    print(f"  Question dataset:       {question_dataset}")
    print(f"  Question verifier out:  {question_results}")
    print(f"  Question metrics:       {question_metrics_output}")
    if args.run_quality_audit:
        print(f"Quality audit: {quality_output}")


if __name__ == "__main__":
    main()
