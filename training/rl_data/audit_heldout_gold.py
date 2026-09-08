#!/usr/bin/env python3
"""Audit heldout gold labels with the terra cascade + DISAGREE protocol.

Does not filter visual/multi-question stems: the goal is to flag untrustworthy
gold on the actual 88-item gate set.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from training.rl_data.generate_sft_solutions import (
    DEFAULT_FEWSHOT,
    TokenBudget,
    cascade_for_row,
    load_fewshot,
)
from training.rl_data.screen_training_data import question_text, sample_id


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _gold_text(row: Dict[str, Any]) -> str:
    lab = row.get("label") if row.get("label") is not None else row.get("solution")
    if isinstance(lab, list):
        return " ; ".join(str(x) for x in lab if str(x).strip())
    return str(lab or "")


def heldout_to_prompt(row: Dict[str, Any]) -> Dict[str, Any]:
    q = question_text(row)
    msgs = row.get("input") if isinstance(row.get("input"), list) else row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        msgs = [
            {
                "role": "system",
                "content": (
                    "You are an expert physics competition solver. "
                    "Show clear step-by-step reasoning and put the final answer in \\boxed{}."
                ),
            },
            {"role": "user", "content": q},
        ]
    meta = row.get("metadata") or {}
    return {
        "messages": [m for m in msgs if isinstance(m, dict) and m.get("role") != "assistant"],
        "question": q,
        "solution": _gold_text(row),
        "sample_id": sample_id(row),
        "source": meta.get("source") or row.get("source") or "heldout_eval",
        "label": row.get("label"),
    }


def write_trusted_heldout(heldout_path: Path, disagree_ids: List[str], out_path: Path) -> Dict[str, Any]:
    drop = {str(x) for x in disagree_ids if x}
    kept: List[Dict[str, Any]] = []
    n_drop = 0
    for row in _load_jsonl(heldout_path):
        sid = sample_id(row)
        if sid and sid in drop:
            n_drop += 1
            continue
        kept.append(row)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"n_in": n_drop + len(kept), "n_out": len(kept), "n_dropped": n_drop, "output": str(out_path)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--heldout", type=Path, default=ROOT / "data/rl/heldout_eval.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/rl/heldout_gold_audit.jsonl")
    p.add_argument("--report", type=Path, default=ROOT / "data/rl/heldout_gold_audit.json")
    p.add_argument("--disagree", type=Path, default=ROOT / "data/rl/heldout_gold_disagree.jsonl")
    p.add_argument("--api-base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--api-model", default=os.environ.get("SFT_API_MODEL", "gpt-5.6-terra"))
    p.add_argument("--fewshot", type=Path, default=DEFAULT_FEWSHOT)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-total-tokens", type=int, default=0)
    args = p.parse_args()
    if not args.api_base_url or not args.api_key:
        raise SystemExit("heldout gold audit requires OPENAI_BASE_URL and OPENAI_API_KEY")

    fewshot = load_fewshot(args.fewshot)
    rows = [heldout_to_prompt(r) for r in _load_jsonl(args.heldout)]
    done = {str(r.get("sample_id") or "") for r in _load_jsonl(args.output)} if args.output.is_file() else set()
    todo = [r for r in rows if str(r.get("sample_id") or "") not in done]
    client = OpenAI(base_url=args.api_base_url, api_key=args.api_key, timeout=args.timeout)
    budget = TokenBudget(args.max_total_tokens)
    stop = threading.Event()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.disagree.parent.mkdir(parents=True, exist_ok=True)

    def work(row: Dict[str, Any]) -> Dict[str, Any]:
        text, verify = cascade_for_row(
            client,
            args.api_model,
            row,
            args.max_tokens,
            args.timeout,
            fewshot=fewshot,
            stop=stop,
            budget=budget,
            min_chars=400,
        )
        rec = {
            "sample_id": row.get("sample_id"),
            "source": row.get("source"),
            "question": row.get("question"),
            "solution": row.get("solution"),
            "verify": verify,
            "text": text or "",
        }
        return rec

    n_new = 0
    with args.output.open("a", encoding="utf-8") as out, args.disagree.open("a", encoding="utf-8") as dfile:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futs = [pool.submit(work, row) for row in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                if rec.get("verify") == "disagree":
                    dfile.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    dfile.flush()
                n_new += 1
                if i % 10 == 0:
                    print(f"[heldout-audit] {i}/{len(todo)}", flush=True)

    all_recs = _load_jsonl(args.output)
    counts: Dict[str, int] = {}
    for rec in all_recs:
        counts[str(rec.get("verify") or "unknown")] = counts.get(str(rec.get("verify") or "unknown"), 0) + 1
    n = max(len(all_recs), 1)
    report = {
        "n": len(all_recs),
        "n_new": n_new,
        "counts": counts,
        "disagree_rate": counts.get("disagree", 0) / n,
        "output": str(args.output),
        "disagree": str(args.disagree),
        "drop_if_near_30pct": counts.get("disagree", 0) / n >= 0.25,
        "disagree_ids": [r.get("sample_id") for r in all_recs if r.get("verify") == "disagree"],
    }
    if report["drop_if_near_30pct"] and report["disagree_ids"]:
        trusted = ROOT / "data/rl/heldout_eval_trusted.jsonl"
        report["trusted"] = write_trusted_heldout(args.heldout, report["disagree_ids"], trusted)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
