#!/usr/bin/env python3
"""Watch GRPO logs vs reward metrics; flag hacking and hard-stop training.

One-shot recap (legacy):
  --logging-jsonl --output [--heldout-acc-json] [--recap-json] [--min-std]

Live fail-closed monitor:
  --metrics --train-log [--pid-file] [--poll-sec] [--once]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

STOP_ZERO_STD_STEPS = 3
LENGTH_EXPLODE_RATIO = 1.8
ACC_DROP_STEPS = 10

HARD_STOP = {
    "zero_std_three_steps",
    "reward_up_length_explosion",
    "nan_loss",
    "bad_grad_norm",
    "answer_acc_declining",
}


def _step_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if "log_history" in obj:
            continue
        if "reward" in obj:
            rows.append(obj)
    return rows


def _tail_jsonl(path: Path, n: int = 400) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]
    rows: List[Dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _extract_train_rows(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-400:]:
        if "{" not in line:
            continue
        start = line.find("{")
        try:
            obj = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and (
            "loss" in obj or "reward" in obj or "grad_norm" in obj or "rewards/mean" in obj
        ):
            rows.append(obj)
    return rows


def _group_std(row: Dict[str, Any]) -> float:
    for key in (
        "physics_reward_group_std_mean",
        "physics_llm_step_group_std_mean",
        "reward_std",
        "group_std",
    ):
        if row.get(key) is not None:
            return float(row[key])
    return 1.0


def evaluate(metrics_rows: List[Dict[str, Any]], train_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Fail-closed checks used by the live monitor."""
    reasons: List[str] = []
    warnings: List[str] = []
    mixed = [float(r["physics_mixed_acc_group_rate"]) for r in metrics_rows if r.get("physics_mixed_acc_group_rate") is not None]
    accs = [float(r["physics_answer_acc"]) for r in metrics_rows if r.get("physics_answer_acc") is not None]
    zero_run = 0
    for row in metrics_rows[-STOP_ZERO_STD_STEPS:]:
        if _group_std(row) <= 1e-12:
            zero_run += 1
        else:
            zero_run = 0
    if len(metrics_rows) >= STOP_ZERO_STD_STEPS and zero_run >= STOP_ZERO_STD_STEPS:
        reasons.append("zero_std_three_steps")
    lengths: List[float] = []
    rewards: List[float] = []
    for row in train_rows:
        if row.get("completion_length") is not None:
            lengths.append(float(row["completion_length"]))
        if row.get("reward") is not None:
            rewards.append(float(row["reward"]))
        loss = row.get("loss")
        if loss is not None:
            try:
                if not (float("-inf") < float(loss) < float("inf")):
                    reasons.append("nan_loss")
            except (TypeError, ValueError):
                reasons.append("nan_loss")
        gn = row.get("grad_norm")
        if gn is not None:
            try:
                gnf = float(gn)
                if not (gnf == gnf) or gnf > 1e4:
                    reasons.append("bad_grad_norm")
            except (TypeError, ValueError):
                reasons.append("bad_grad_norm")
    if len(lengths) >= 8 and len(rewards) >= 8:
        early_len = sum(lengths[:4]) / 4.0
        late_len = sum(lengths[-4:]) / 4.0
        early_r = sum(rewards[:4]) / 4.0
        late_r = sum(rewards[-4:]) / 4.0
        if late_r > early_r + 0.05 and early_len > 0 and late_len / early_len >= LENGTH_EXPLODE_RATIO:
            reasons.append("reward_up_length_explosion")
    elif len(lengths) >= 8:
        early_len = sum(lengths[:4]) / 4.0
        late_len = sum(lengths[-4:]) / 4.0
        if early_len > 0 and late_len / early_len >= LENGTH_EXPLODE_RATIO:
            reasons.append("reward_up_length_explosion")
    if len(accs) >= ACC_DROP_STEPS and max(accs[:5]) > 0.01:
        last = accs[-ACC_DROP_STEPS:]
        declines = sum(1 for i in range(len(last) - 1) if last[i + 1] < last[i] - 1e-9)
        if declines >= ACC_DROP_STEPS - 2 and last[-1] < last[0] - 1e-6:
            reasons.append("answer_acc_declining")
    if mixed and mixed[-1] < 0.15:
        warnings.append("low_mixed_group_rate")
    truncs: List[float] = []
    for row in list(metrics_rows) + list(train_rows):
        for key in (
            "physics_trunc_rate",
            "physics_overlong_rate",
            "trunc_rate",
            "completions/clipped_ratio",
            "clipped_ratio",
        ):
            if row.get(key) is not None:
                truncs.append(float(row[key]))
                break
    if len(truncs) >= 6:
        early_t = sum(truncs[:3]) / 3.0
        late_t = sum(truncs[-3:]) / 3.0
        if late_t >= 0.08 and (early_t <= 1e-9 or late_t / max(early_t, 1e-9) >= 1.5):
            warnings.append("truncation_rate_rising")
    hard = sorted(set(reasons) & HARD_STOP)
    return {
        "stop": bool(hard),
        "reasons": hard,
        "warnings": sorted(set(warnings)),
        "n_metrics": len(metrics_rows),
        "n_train": len(train_rows),
        "physics_mixed_acc_group_rate": mixed[-1] if mixed else None,
        "physics_answer_acc": accs[-1] if accs else None,
    }


def maybe_stop_training(pid_file: Path, report: Dict[str, Any]) -> None:
    if not report.get("stop"):
        return
    if not pid_file.is_file():
        return
    pid = pid_file.read_text(encoding="utf-8").strip()
    if not pid.isdigit():
        return
    Path(str(pid_file) + ".stop_reason.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    try:
        os.kill(int(pid), 15)
    except OSError:
        pass


def summarize_logging(
    logging_jsonl: Path,
    *,
    heldout_acc_json: Path | None = None,
    recap_json: Path | None = None,
    min_std: float = 0.12,
) -> Dict[str, Any]:
    steps = _step_rows(logging_jsonl)
    rewards = [float(s["reward"]) for s in steps]
    stds = [float(s.get("reward_std") or 0.0) for s in steps]
    clipped = [float(s.get("completions/clipped_ratio") or 0.0) for s in steps]
    report: Dict[str, Any] = {
        "n_steps": len(steps),
        "reward_mean": sum(rewards) / max(len(rewards), 1),
        "reward_std_mean": sum(stds) / max(len(stds), 1),
        "clipped_ratio_mean": sum(clipped) / max(len(clipped), 1),
        "flags": [],
    }
    if stds and (sum(stds) / len(stds)) < min_std:
        report["flags"].append("low_ingroup_std")
    if clipped and (sum(clipped) / len(clipped)) > 0.5:
        report["flags"].append("high_clip_ratio")
    if heldout_acc_json and heldout_acc_json.is_file():
        acc = json.loads(heldout_acc_json.read_text(encoding="utf-8"))
        report["heldout_answer_acc"] = acc.get("answer_acc")
    if recap_json and recap_json.is_file():
        recap = json.loads(recap_json.read_text(encoding="utf-8"))
        report["recap_30b_mean"] = recap.get("mean_b") or recap.get("mean_a")
        report["spearman_vs_30b"] = recap.get("spearman")
        if recap.get("spearman") is not None and recap["spearman"] < 0.2:
            report["flags"].append("self_judge_decoupled_from_30b")
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--logging-jsonl", type=Path, default=None)
    p.add_argument("--heldout-acc-json", type=Path, default=None)
    p.add_argument("--recap-json", type=Path, default=None, help="optional 30B smoke_self_judge.json")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--min-std", type=float, default=0.12)
    p.add_argument("--metrics", type=Path, default=None)
    p.add_argument("--train-log", type=Path, default=None)
    p.add_argument("--pid-file", type=Path, default=None)
    p.add_argument("--poll-sec", type=float, default=30)
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    if args.metrics and args.train_log:
        while True:
            metrics = _tail_jsonl(args.metrics)
            train = _extract_train_rows(args.train_log)
            report = evaluate(metrics, train)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if args.pid_file and report.get("n_train", 0) >= 1:
                maybe_stop_training(args.pid_file, report)
            if args.once or report.get("stop"):
                return 2 if report.get("stop") else 0
            time.sleep(args.poll_sec)

    if not args.logging_jsonl or not args.output:
        p.error("need --logging-jsonl and --output, or --metrics and --train-log")
    report = summarize_logging(
        args.logging_jsonl,
        heldout_acc_json=args.heldout_acc_json,
        recap_json=args.recap_json,
        min_std=args.min_std,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
