#!/usr/bin/env python3
"""Search local disks for a missing outcome-only GRPO run and write an audit.

The Sep 7 waiter on this host never launched. If the run lived on another
machine, drop its artifacts under one of the search roots (or pass --extra)
and re-run this script.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]

SEARCH_ROOTS = [
    Path("/slow_share/jinjianhan/ckpt"),
    ROOT / "logs",
]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _scan_logging(path: Path) -> Dict[str, Any]:
    n = 0
    last: Dict[str, Any] = {}
    kl_max = 0.0
    clip_sum = 0.0
    clip_n = 0
    reward_first = None
    reward_last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "reward" not in row:
                continue
            n += 1
            last = row
            if reward_first is None:
                reward_first = float(row["reward"])
            reward_last = float(row["reward"])
            if row.get("kl") is not None:
                kl_max = max(kl_max, float(row["kl"]))
            if row.get("completions/clipped_ratio") is not None:
                clip_sum += float(row["completions/clipped_ratio"])
                clip_n += 1
    return {
        "n_reward_rows": n,
        "reward_first": reward_first,
        "reward_last": reward_last,
        "kl_max": kl_max,
        "clipped_ratio_mean": (clip_sum / clip_n) if clip_n else None,
        "last_keys": sorted(last.keys())[:20],
        "epoch_last": last.get("epoch"),
        "max_completion_hint": last.get("completions/max_length"),
    }


def _walk_limited(root: Path, names: tuple[str, ...], max_depth: int = 5) -> List[Path]:
    found: List[Path] = []
    if not root.exists():
        return found
    root_s = str(root.resolve())
    skip = {".git", "node_modules", "__pycache__", "hf_cache", "datasets", "venv"}
    for dirpath, dirnames, filenames in os.walk(root_s):
        rel = os.path.relpath(dirpath, root_s)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dirnames[:] = [d for d in dirnames if d not in skip and not d.startswith(".")]
        if depth >= max_depth:
            dirnames[:] = []
        for name in filenames:
            if name in names:
                found.append(Path(dirpath) / name)
    return found


def search(roots: List[Path]) -> Dict[str, Any]:
    hits: List[Dict[str, Any]] = []
    names = {
        "swift_launch_report.json",
        "logging.jsonl",
        "heldout_scores.json",
        "gate.json",
        "physics_reward_metrics.jsonl",
    }
    candidates = [
        Path("/slow_share/jinjianhan/ckpt/qwen3-8b-outcome-only-rl"),
        Path("/slow_share/jinjianhan/ckpt/qwen3-8b-outcome-only-smoke"),
        Path("/slow_share/jinjianhan/ckpt/qwen3-8b-outcome-only-pilot"),
        Path("/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-rl"),
        Path("/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-pilot10"),
        ROOT / "logs",
    ]
    extra_dirs = [r for r in roots if r not in candidates]
    for folder in candidates + extra_dirs:
        if not folder.exists():
            continue
        if folder.is_file():
            paths = [folder]
        else:
            paths = [p for p in folder.glob("*") if p.name in names]
            paths += list(folder.glob("v*/logging.jsonl"))
            paths += list(folder.glob("v*/swift_launch_report.json"))
        for path in paths:
            if not path.is_file():
                continue
            rel = str(path)
            name = path.name
            rec: Dict[str, Any] = {"path": rel, "name": name, "mtime": path.stat().st_mtime}
            if name == "swift_launch_report.json":
                rec["json"] = _load_json(path)
            elif name == "logging.jsonl" and path.stat().st_size < 20_000_000:
                rec["summary"] = _scan_logging(path)
            elif name.endswith(".json"):
                rec["json"] = _load_json(path)
            hits.append(rec)
    local = {
        "outcome_ckpt_exists": Path("/slow_share/jinjianhan/ckpt/qwen3-8b-outcome-only-rl").exists(),
        "hybrid_rl_dir": str(Path("/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-rl")),
        "hybrid_rl_contents": (
            [p.name for p in Path("/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-rl").iterdir()]
            if Path("/slow_share/jinjianhan/ckpt/qwen3-8b-hybrid-outcome-rl").exists()
            else []
        ),
        "waiter_log": str(ROOT / "logs/wait_and_launch_outcome_grpo.log"),
        "waiter_log_exists": (ROOT / "logs/wait_and_launch_outcome_grpo.log").is_file(),
    }
    waiter_tail = ""
    waiter = ROOT / "logs/wait_and_launch_outcome_grpo.log"
    if waiter.is_file():
        waiter_tail = waiter.read_text(encoding="utf-8", errors="replace")[-1500:]
    return {
        "at": datetime.now(timezone.utc).isoformat(),
        "local": local,
        "waiter_tail": waiter_tail,
        "hits": hits,
        "d1_code_is_binary_any_of": True,
        "d1_note": "_check_answer historically returned on first matching gold part (any-of, binary).",
        "d2_onset_kl_max": 0.0011,
        "d3_onset_clipped_ratio_mean": 0.48,
        "prediction": (
            "If the other-machine run used launch_hybrid_grpo_4gpu.sh MODE=full "
            "before this overhaul, it inherited max_completion_length=3072, "
            "MAX_STEPS=30, binary any-of grading, and the 65-item heldout."
        ),
    }


def render_md(audit: Dict[str, Any]) -> str:
    local = audit["local"]
    lines = [
        "# Outcome-only GRPO 审计（P0）",
        "",
        f"生成时间：`{audit['at']}`",
        "",
        "## 本机",
        "",
        f"- `qwen3-8b-outcome-only-rl` 存在：`{local['outcome_ckpt_exists']}`",
        f"- `qwen3-8b-hybrid-outcome-rl` 内容：`{local['hybrid_rl_contents']}`",
        f"- waiter 日志存在：`{local['waiter_log_exists']}`",
        "",
        "waiter 尾部：",
        "",
        "```",
        str(audit.get("waiter_tail") or "(empty)").rstrip(),
        "```",
        "",
        "## 缺陷对照（共享代码，与机器无关）",
        "",
        "| 缺陷 | 本机证据 | 他机预判 |",
        "|---|---|---|",
        "| D1 奖励非同构 | `_check_answer` 二值 any-of | 同一份代码则同样命中 |",
        "| D2 步数不足 | onset 100 步 kl≤0.0011 | 若 MAX_STEPS≤100 则同样 |",
        "| D3 长度预算 | onset clipped_ratio≈0.48 @1536 | full 模式原 3072，评测 8192 |",
        "| D4 测评分辨率 | 65 题 SE=0.054 | 若用 heldout_eval_trusted.jsonl 则同样 |",
        "",
        audit["prediction"],
        "",
        "## 搜索命中",
        "",
    ]
    if not audit["hits"]:
        lines.append("未在本机搜索根下找到 outcome-only 的 `swift_launch_report.json` / `logging.jsonl` / heldout 打分。")
        lines.append("")
        lines.append("请把另一台机器的 checkpoint 目录拷到 `/slow_share/jinjianhan/ckpt/` 后重跑：")
        lines.append("")
        lines.append("```bash")
        lines.append("python training/swift/recover_outcome_run.py --write")
        lines.append("```")
    else:
        for hit in audit["hits"]:
            lines.append(f"- `{hit['path']}`")
            if hit.get("json"):
                slim = {
                    k: hit["json"].get(k)
                    for k in (
                        "phase",
                        "reward_mode",
                        "max_steps",
                        "max_completion_length",
                        "num_generations",
                        "overlong_filter",
                    )
                    if isinstance(hit["json"], dict)
                }
                lines.append(f"  `{json.dumps(slim, ensure_ascii=False)}`")
            if hit.get("summary"):
                lines.append(f"  `{json.dumps(hit['summary'], ensure_ascii=False)}`")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extra", type=Path, action="append", default=[])
    p.add_argument("--write", action="store_true")
    p.add_argument("--out-json", type=Path, default=ROOT / "logs/outcome_rl_p0_audit.json")
    p.add_argument("--out-md", type=Path, default=ROOT / "docs/outcome_rl_audit.md")
    args = p.parse_args()
    roots = list(SEARCH_ROOTS) + list(args.extra)
    audit = search(roots)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md = render_md(audit)
    if args.write:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
