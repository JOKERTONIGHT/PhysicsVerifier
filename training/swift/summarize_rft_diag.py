#!/usr/bin/env python3
"""Locate the RFT/SFT collapse step from per-checkpoint heldout gates.

Finds the first save-step where no_boxed jumps from ~base (0.2%) toward
the previously observed 23% collapse, and where degrade_rate exceeds 0.05.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]


def _step(ckpt: str) -> Optional[int]:
    m = re.search(r"checkpoint-(\d+)", ckpt)
    return int(m.group(1)) if m else None


def summarize_gate(
    report: Dict[str, Any],
    *,
    no_boxed_jump: float = 0.05,
    degrade_jump: float = 0.05,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for rec in report.get("checkpoints") or []:
        ckpt = str(rec.get("ckpt") or "")
        rows.append(
            {
                "step": _step(ckpt),
                "ckpt": ckpt,
                "pass": bool(rec.get("pass")),
                "sft_part_avg_at_k": rec.get("sft_part_avg_at_k"),
                "sft_degrade_rate": rec.get("sft_degrade_rate"),
                "sft_no_boxed_rate": rec.get("sft_no_boxed_rate"),
                "sft_repetition_rate": rec.get("sft_repetition_rate"),
                "error": rec.get("error"),
            }
        )
    rows.sort(key=lambda r: (r["step"] is None, r["step"] or 0))
    first_no_box = next(
        (r for r in rows if float(r.get("sft_no_boxed_rate") or 0) >= no_boxed_jump),
        None,
    )
    first_degrade = next(
        (r for r in rows if float(r.get("sft_degrade_rate") or 0) >= degrade_jump),
        None,
    )
    return {
        "n_checkpoints": len(rows),
        "rows": rows,
        "first_no_boxed_ge_5pct_step": None if first_no_box is None else first_no_box["step"],
        "first_degrade_ge_5pct_step": None if first_degrade is None else first_degrade["step"],
        "max_no_boxed_rate": max((float(r.get("sft_no_boxed_rate") or 0) for r in rows), default=0.0),
        "max_degrade_rate": max((float(r.get("sft_degrade_rate") or 0) for r in rows), default=0.0),
        "any_eval": any(r.get("sft_part_avg_at_k") is not None for r in rows),
    }


def backfill_from_scores(
    ckpt_root: Path,
    base_scores: Path,
    *,
    min_part_avg: float = 0.252,
    max_degrade_rate: float = 0.05,
    max_no_boxed_rate: float = 0.05,
) -> Dict[str, Any]:
    """Write missing gate.json from heldout_scores.json under merged checkpoints."""
    sys.path.insert(0, str(ROOT))
    from evaluation.benchmarks.hipho.score_hipho_predictions import evaluate_gate

    base: Dict[str, Any] = {}
    if base_scores.is_file():
        base = json.loads(base_scores.read_text(encoding="utf-8"))
    checkpoints: List[Dict[str, Any]] = []
    score_files = sorted(ckpt_root.glob("**/heldout_fast_eval/heldout_scores.json"))
    for scores_path in score_files:
        merged = scores_path.parent.parent
        lora = Path(str(merged).removesuffix("-merged")) if str(merged).endswith("-merged") else merged
        sft = json.loads(scores_path.read_text(encoding="utf-8"))
        gate = evaluate_gate(
            sft,
            base,
            min_part_avg=min_part_avg,
            max_degrade_rate=max_degrade_rate,
            max_no_boxed_rate=max_no_boxed_rate,
        )
        gate["ckpt"] = str(lora)
        gate["merged"] = str(merged)
        gate_path = scores_path.parent / "gate.json"
        gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
        checkpoints.append(gate)
    checkpoints.sort(key=lambda g: (_step(str(g.get("ckpt") or "")) is None, _step(str(g.get("ckpt") or "")) or 0))
    report = {
        "ckpt_root": str(ckpt_root),
        "checkpoints": checkpoints,
        "any_pass": any(bool(c.get("pass")) for c in checkpoints),
        "backfilled": True,
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--backfill-ckpt-root", type=Path, default=None)
    p.add_argument("--base-scores", type=Path, default=ROOT / "results/hipho_baseline_matrix_8b/base_8b_h88/heldout_scores.json")
    args = p.parse_args()
    if args.backfill_ckpt_root:
        report = backfill_from_scores(args.backfill_ckpt_root, args.base_scores)
        summary_path = args.summary or (args.backfill_ckpt_root / "gate_summary.json")
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        collapse = summarize_gate(report)
        collapse_path = summary_path.with_name("collapse_report.json")
        collapse_path.write_text(json.dumps(collapse, indent=2) + "\n", encoding="utf-8")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(collapse, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(collapse, indent=2))
        return 0
    if not args.summary:
        p.error("--summary is required unless --backfill-ckpt-root is set")
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    collapse = summarize_gate(report)
    text = json.dumps(collapse, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
