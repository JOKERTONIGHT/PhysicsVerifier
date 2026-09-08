#!/usr/bin/env python3
"""Sample n completions per Swift prompt and write acc-labelled rollouts."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from evaluation.benchmarks.hipho.score_hipho_predictions import score_prediction


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _messages(row: Dict[str, Any]) -> List[Dict[str, str]]:
    for key in ("messages", "input"):
        msgs = row.get(key)
        if isinstance(msgs, list) and msgs:
            out = []
            for m in msgs:
                if not isinstance(m, dict):
                    continue
                role = str(m.get("role") or "user")
                if role == "assistant":
                    continue
                out.append({"role": role, "content": str(m.get("content") or "")})
            if out:
                return out
    q = str(row.get("question") or "")
    return [{"role": "user", "content": q}] if q else []


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=ROOT / "data/rl/swift_prompts_max2048.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-url", default=os.environ.get("PASSRATE_BASE_URL", "http://127.0.0.1:8766/v1"))
    p.add_argument("--model", default=os.environ.get("PASSRATE_MODEL", "qwen3-8b"))
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--max-prompts", type=int, default=0)
    p.add_argument("--summary", type=Path, default=None)
    args = p.parse_args()

    rows = _load_jsonl(args.prompts)
    if args.max_prompts:
        rows = rows[: args.max_prompts]
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def work(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        messages = _messages(row)
        gold = row.get("solution") or row.get("label")
        labels = gold if isinstance(gold, list) else ([str(gold)] if gold else [])
        for i in range(args.n_samples):
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            text = (resp.choices[0].message.content or "") if resp.choices else ""
            metrics = score_prediction(text, labels)
            rec = {
                "sample_id": row.get("sample_id"),
                "question": row.get("question"),
                "solution": gold,
                "response": text,
                "rollout_index": i,
                "acc": bool(metrics["item_correct"]),
                "part_frac": metrics["part_frac"],
                "no_boxed": metrics["no_boxed"],
                "repetitive": metrics["repetitive"],
            }
            out.append(rec)
        return out

    written = 0
    with args.output.open("w", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futs = [pool.submit(work, row) for row in rows]
            for i, fut in enumerate(as_completed(futs), 1):
                for rec in fut.result():
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                if i % 20 == 0:
                    print(f"[pass-rate] prompts {i}/{len(rows)} rows={written}", flush=True)
    print(json.dumps({"prompts": len(rows), "records": written, "output": str(args.output)}))
    if args.summary:
        from collections import defaultdict

        recs = _load_jsonl(args.output)
        by_id: Dict[str, List[bool]] = defaultdict(list)
        for rec in recs:
            sid = str(rec.get("sample_id") or "")
            by_id[sid].append(bool(rec.get("acc")))
        n_ids = max(len(by_id), 1)
        pass1 = sum(v[0] for v in by_id.values() if v) / n_ids
        passk = sum(1 for v in by_id.values() if any(v)) / n_ids
        n_any = sum(1 for v in by_id.values() if any(v))
        summary = {
            "n_prompts": len(by_id),
            "n_samples": args.n_samples,
            "pass@1": pass1,
            "pass@k": passk,
            "n_pass_at_k": n_any,
            "output": str(args.output),
        }
        args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
