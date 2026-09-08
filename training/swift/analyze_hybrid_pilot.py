#!/usr/bin/env python3
"""Summarize a hybrid GRPO pilot: boxed rate, mixed-acc groups, clip, heldout-50."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _train_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "{" not in line:
            continue
        start = line.find("{")
        try:
            obj = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("reward" in obj or "loss" in obj or "completions/clipped_ratio" in obj):
            rows.append(obj)
    return rows


def _mean(vals: List[float]) -> float:
    return sum(vals) / max(len(vals), 1)


def evaluate_pilot(
    *,
    metrics: List[Dict[str, Any]],
    train: List[Dict[str, Any]],
    heldout: Optional[Dict[str, Any]],
    base_heldout_acc: float = 0.0267,
) -> Dict[str, Any]:
    accs = [float(r["physics_answer_acc"]) for r in metrics if r.get("physics_answer_acc") is not None]
    mixed = [float(r["physics_mixed_acc_group_rate"]) for r in metrics if r.get("physics_mixed_acc_group_rate") is not None]
    fmt = [float(r["physics_format_rate"]) for r in metrics if r.get("physics_format_rate") is not None]
    clipped = [float(r["completions/clipped_ratio"]) for r in train if r.get("completions/clipped_ratio") is not None]
    stds = [float(r.get("reward_std") or r.get("physics_hybrid_zero_std_rate") or 0.0) for r in train]
    heldout_acc = None
    if heldout:
        heldout_acc = float(heldout.get("answer_acc") or heldout.get("boxed_acc") or 0.0)
    flags: List[str] = []
    healthy: List[str] = []
    clip_mean = _mean(clipped[-5:] if clipped else [])
    if clip_mean >= 0.6:
        flags.append("high_clip_ratio")
    elif clipped:
        healthy.append("clip_ok")
    fmt_mean = _mean(fmt[-5:] if fmt else [])
    fmt_early = _mean(fmt[:3]) if len(fmt) >= 3 else fmt_mean
    if fmt and fmt_mean + 0.05 < fmt_early:
        flags.append("boxed_rate_dropped")
    elif fmt:
        healthy.append("boxed_stable")
    acc_mean = _mean(accs[-5:] if accs else [])
    mixed_mean = _mean(mixed[-5:] if mixed else [])
    if mixed_mean > 0:
        healthy.append("mixed_acc_groups")
    if accs and acc_mean <= 1e-9 and not mixed:
        flags.append("answer_acc_always_zero")
    if heldout_acc is not None and heldout_acc + 0.02 < base_heldout_acc:
        flags.append("heldout_regressed")
    elif heldout_acc is not None:
        healthy.append("heldout_not_worse")
    verdict = "pass"
    if "boxed_rate_dropped" in flags or "heldout_regressed" in flags:
        verdict = "fail"
    elif "high_clip_ratio" in flags and "mixed_acc_groups" not in healthy:
        verdict = "watch"
    report = {
        "verdict": verdict,
        "n_metric_rows": len(metrics),
        "n_train_rows": len(train),
        "physics_answer_acc_mean": acc_mean,
        "physics_mixed_acc_group_rate_mean": mixed_mean,
        "physics_format_rate_mean": fmt_mean,
        "clipped_ratio_mean": clip_mean,
        "heldout_acc": heldout_acc,
        "base_heldout_acc": base_heldout_acc,
        "flags": flags,
        "healthy": healthy,
        "note": "Train reward is ignored. Phase-1 pass = boxed not collapsing and correct>incorrect.",
    }
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=Path, default=Path("/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-pilot10"))
    p.add_argument("--metrics", type=Path, default=Path("/home/jinjianhan/PhysicsVerifier/logs/physics_reward_metrics.jsonl"))
    p.add_argument("--heldout-scores", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()
    train_log = args.ckpt / "swift_grpo.log"
    logs = sorted(args.ckpt.glob("v*/logging.jsonl"), key=lambda x: x.stat().st_mtime)
    train_rows = _train_rows(logs[-1] if logs else train_log)
    if not train_rows:
        train_rows = _train_rows(train_log)
    held_path = args.heldout_scores or (args.ckpt / "heldout_fast_eval" / "heldout_scores.json")
    heldout = _load_json(held_path) if held_path else None
    report = evaluate_pilot(
        metrics=_metric_rows(args.metrics),
        train=train_rows,
        heldout=heldout,
    )
    out = args.output or (args.ckpt / "pilot_observe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["verdict"] != "fail" else 3


if __name__ == "__main__":
    raise SystemExit(main())
