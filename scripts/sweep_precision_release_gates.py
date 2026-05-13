from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parents[1]


def _parse_csv_floats(raw: str) -> List[float]:
    out: List[float] = []
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


def _parse_csv_modes(raw: str) -> List[str]:
    allowed = {"strict", "balanced", "score_only"}
    modes = [part.strip() for part in str(raw or "").split(",") if part.strip()]
    bad = [mode for mode in modes if mode not in allowed]
    if bad:
        raise SystemExit(f"Unsupported precision mode(s): {', '.join(bad)}")
    return modes or ["strict"]


def _run(cmd: List[str]) -> None:
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run strict/balanced/score-only precision sweeps and compute error-level metrics."
    )
    parser.add_argument("--input", required=True, help="Verifier input samples JSON.")
    parser.add_argument("--dataset", required=True, help="GT error-level dataset JSON.")
    parser.add_argument("--unified-catalog", required=True, help="Unified v2 catalog path.")
    parser.add_argument("--output-dir", required=True, help="Directory for sweep outputs.")
    parser.add_argument("--model", default="qwen3-30b-a3b")
    parser.add_argument("--modes", default="strict,balanced,score_only")
    parser.add_argument("--thresholds", default="4,6,8,10")
    parser.add_argument(
        "--no-symbolic-check",
        action="store_true",
        help="Disable the experience-code symbolic verification (on by default).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modes = _parse_csv_modes(args.modes)
    thresholds = _parse_csv_floats(args.thresholds)
    summaries = []

    for mode in modes:
        for threshold in thresholds:
            label = f"{mode}_score_ge_{str(threshold).replace('.', '_')}"
            pred_path = out_dir / f"pred_{label}.json"
            audit_path = out_dir / f"audit_{label}.json"
            full_path = out_dir / f"full_{label}.json"
            metrics_path = out_dir / f"metrics_{label}.json"

            verifier_cmd = [
                sys.executable,
                "scripts/run_verifier.py",
                "--input",
                args.input,
                "--output",
                str(pred_path),
                "--symbolic-output",
                str(audit_path),
                "--full-output",
                str(full_path),
                "--unified-catalog",
                args.unified_catalog,
                "--model",
                args.model,
                "--precision-mode",
                mode,
                "--min-diagnostic-rule-score",
                str(threshold),
            ]
            if args.no_symbolic_check:
                verifier_cmd.append("--no-symbolic-check")
            _run(verifier_cmd)

            _run(
                [
                    sys.executable,
                    "scripts/evaluate_physics_eval_sets.py",
                    "--dataset",
                    args.dataset,
                    "--results",
                    str(pred_path),
                    "--audit",
                    str(audit_path),
                    "--output",
                    str(metrics_path),
                ]
            )

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary = dict(metrics.get("summary") or {})
            summary.update({"mode": mode, "threshold": threshold, "label": label})
            summaries.append(summary)

    (out_dir / "precision_sweep_summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
