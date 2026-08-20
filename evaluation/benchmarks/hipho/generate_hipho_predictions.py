#!/usr/bin/env python3
"""Generate model predictions for HiPhO text-only subset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

ROOT = Path(__file__).resolve().parents[3]
SYSTEM = (
    "You are an expert physics olympiad solver. "
    "Provide rigorous step-by-step reasoning and final answers in \\boxed{}."
)


def _question_from_row(row: Dict[str, Any]) -> str:
    question = row.get("question") or row.get("problem")
    if question:
        return str(question)
    inp = row.get("input")
    if isinstance(inp, list):
        parts = []
        for msg in inp:
            if isinstance(msg, dict) and msg.get("content"):
                parts.append(str(msg["content"]))
        if parts:
            return "\n".join(parts)
    if isinstance(inp, str) and inp.strip():
        return inp
    meta = row.get("metadata") or {}
    if meta.get("question"):
        return str(meta["question"])
    return ""


def _messages_from_row(row: Dict[str, Any]) -> List[Dict[str, str]]:
    inp = row.get("input")
    if isinstance(inp, list) and inp:
        messages = []
        for msg in inp:
            if isinstance(msg, dict) and msg.get("content"):
                messages.append({"role": str(msg.get("role", "user")), "content": str(msg["content"])})
        if messages:
            return messages
    question = _question_from_row(row)
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8766/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "qwen3-30b-a3b-instruct-2507"))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--start-index", type=int, default=0, help="Skip first N input rows (resume)")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output line count")
    args = parser.parse_args()

    rows = _load_jsonl(args.input)
    if args.max_samples:
        rows = rows[: args.max_samples]

    start_idx = max(args.start_index, 0)
    if args.resume and args.output.is_file():
        with args.output.open("r", encoding="utf-8") as existing:
            start_idx = max(start_idx, sum(1 for line in existing if line.strip()))

    rows = rows[start_idx:]
    client = OpenAI(base_url=args.base_url, api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if start_idx > 0 and args.output.is_file() else "w"
    with args.output.open(mode, encoding="utf-8") as out:
        for row in rows:
            messages = _messages_from_row(row)
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            prediction = resp.choices[0].message.content or ""
            rec = dict(row)
            rec["prediction"] = prediction
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()

    print(
        json.dumps(
            {
                "count": len(rows),
                "start_index": start_idx,
                "output": str(args.output),
                "mode": mode,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
