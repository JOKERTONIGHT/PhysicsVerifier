from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def _run(cmd: str) -> None:
    print(f"[RUN] {cmd}")
    res = subprocess.run(cmd, shell=True)
    if res.returncode != 0:
        raise SystemExit(f"Command failed ({res.returncode}): {cmd}")


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end small-scale physics evaluation pipeline.")
    parser.add_argument("--python", type=str, default="./.venv/bin/python")
    parser.add_argument("--input", type=str, default="data/physics_rubric_data_1000.json")
    parser.add_argument("--recall-size", type=int, default=20)
    parser.add_argument("--precision-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--strong-model", type=str, default="gemini3_pro_preview")
    parser.add_argument("--check-model", type=str, default="qwen3-30b-a3b")
    parser.add_argument("--output-dir", type=str, default="results/eval_pipeline")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    recall_dataset = outdir / "evaluation_recall_20.json"
    precision_dataset = outdir / "evaluation_precision_20.json"

    recall_results = outdir / "recall_top_down_results.json"
    recall_audit = outdir / "recall_symbolic_audit.json"
    precision_results = outdir / "precision_top_down_results.json"
    precision_audit = outdir / "precision_symbolic_audit.json"
    metrics_output = outdir / "metrics.json"

    py = shlex.quote(args.python)

    if not args.skip_build:
        _run(
            f"{py} scripts/build_physics_eval_sets.py "
            f"--input {shlex.quote(args.input)} "
            f"--recall-output {shlex.quote(str(recall_dataset))} "
            f"--precision-output {shlex.quote(str(precision_dataset))} "
            f"--recall-size {args.recall_size} "
            f"--precision-size {args.precision_size} "
            f"--seed {args.seed} "
            f"--strong-model {shlex.quote(args.strong_model)}"
        )

    if not args.skip_run:
        _run(
            f"{py} scripts/run_top_down.py "
            f"--input {shlex.quote(str(recall_dataset))} "
            f"--output {shlex.quote(str(recall_results))} "
            f"--symbolic-output {shlex.quote(str(recall_audit))} "
            f"--model {shlex.quote(args.check_model)} "
            f"--unified-catalog catalogs/rules_unified.json"
        )
        _run(
            f"{py} scripts/run_top_down.py "
            f"--input {shlex.quote(str(precision_dataset))} "
            f"--output {shlex.quote(str(precision_results))} "
            f"--symbolic-output {shlex.quote(str(precision_audit))} "
            f"--model {shlex.quote(args.check_model)} "
            f"--unified-catalog catalogs/rules_unified.json"
        )

    _run(
        f"{py} scripts/evaluate_physics_eval_sets.py "
        f"--recall-dataset {shlex.quote(str(recall_dataset))} "
        f"--precision-dataset {shlex.quote(str(precision_dataset))} "
        f"--recall-results {shlex.quote(str(recall_results))} "
        f"--recall-audit {shlex.quote(str(recall_audit))} "
        f"--precision-results {shlex.quote(str(precision_results))} "
        f"--precision-audit {shlex.quote(str(precision_audit))} "
        f"--output {shlex.quote(str(metrics_output))} "
        f"--semantic-match-model {shlex.quote(args.strong_model)}"
    )

    print("Done.")
    print(f"Recall dataset: {recall_dataset}")
    print(f"Precision dataset: {precision_dataset}")
    print(f"Metrics: {metrics_output}")


if __name__ == "__main__":
    main()
