from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
from pathlib import Path
from typing import List


KEEP_PATTERNS = [
    "results/rule_experience_bank.json",
    "results/eval_pipeline/**",
    "results/eval_pipeline_v0409/**",
    "results/scale_curve/**",
    "data/physics_rubric_data_1000.json",
    "data/evaluation_sample_3_physics_error_examples.json",
]

REMOVE_PATTERNS = [
    "logs/**",
    "results/*.json",
    "results/*/*.json",
    "results/*/*.csv",
    "results/*/*.png",
    "data/checkpoints/**",
]


def _match_any(path: str, patterns: List[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def _collect_targets(root: Path) -> List[Path]:
    targets: List[Path] = []

    # Remove project-level __pycache__ (exclude .venv)
    for p in root.rglob("__pycache__"):
        rp = p.relative_to(root).as_posix()
        if rp.startswith(".venv/"):
            continue
        targets.append(p)

    for p in root.rglob("*"):
        if not p.exists() or p.is_dir():
            continue
        rp = p.relative_to(root).as_posix()
        if rp.startswith(".venv/"):
            continue
        if _match_any(rp, REMOVE_PATTERNS) and not _match_any(rp, KEEP_PATTERNS):
            targets.append(p)

    # Deduplicate while preserving order
    out: List[Path] = []
    seen = set()
    for t in targets:
        k = str(t.resolve())
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Cleanup non-core generated artifacts with a safe keep-list.")
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=str, default="results/eval_pipeline/cleanup_report.json")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    targets = _collect_targets(root)

    report = {
        "root": str(root),
        "apply": bool(args.apply),
        "delete_count": len(targets),
        "delete_targets": [str(p.relative_to(root).as_posix()) for p in targets],
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.apply:
        for p in targets:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink()
        print(json.dumps({"deleted": len(targets), "report": str(report_path)}, ensure_ascii=False))
    else:
        print(json.dumps({"planned_delete": len(targets), "report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
