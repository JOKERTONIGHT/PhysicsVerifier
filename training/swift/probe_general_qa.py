#!/usr/bin/env python3
"""Short general-domain QA probe: did SFT/RFT wipe general ability?"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

QUESTIONS = [
    {"id": "fact_en", "q": "What is the capital of France? Answer in one short sentence."},
    {"id": "fact_zh", "q": "中国的首都是哪里？请用一句话回答。"},
    {"id": "arith", "q": "Compute 17 × 24. Reply with only the number."},
    {"id": "instr", "q": "只用一个词回答：晴天时天空通常是什么颜色？"},
    {"id": "logic", "q": "如果所有猫都是动物，小花是一只猫，那么小花是动物吗？只回答「是」或「否」。"},
    {"id": "code", "q": "Write a Python function reverse_list(xs) that returns a reversed copy of a list. Code only."},
    {"id": "translate", "q": "Translate into English, one sentence: 光合作用把光能转化为化学能。"},
    {"id": "hamlet", "q": "Who wrote Hamlet? One name only."},
    {"id": "odds", "q": "List three odd positive integers, separated by commas. No explanation."},
    {"id": "photo", "q": "用不超过 30 个汉字解释什么是光合作用。"},
]

MD_HEADING = re.compile(r"(?m)^#{1,6}\s")
LOOP_TAIL = re.compile(r"(.{20,80})\1{3,}", re.DOTALL)


def looks_loop(text: str) -> bool:
    src = text or ""
    if len(src) < 400:
        return False
    tail = src[-200:]
    return (not tail.strip()) or src.count(tail) >= 3 or bool(LOOP_TAIL.search(src[-1200:]))


def chat(tok, q: str) -> str:
    msgs = [{"role": "user", "content": q}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tok.apply_chat_template(msgs, enable_thinking=False, **kwargs)
    except TypeError:
        return tok.apply_chat_template(msgs, **kwargs)


def generate_one(model, tok, q: str, max_new: int) -> str:
    prompt = chat(tok, q)
    batch = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=max_new,
            do_sample=False,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    gen = out[0, batch["input_ids"].shape[1] :]
    return tok.decode(gen, skip_special_tokens=True).strip()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", default=None, help="optional PEFT/LoRA adapter dir")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--label", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-new", type=int, default=256)
    p.add_argument("--ids", default="", help="comma-separated question ids; empty = all")
    args = p.parse_args()
    items = QUESTIONS
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [x for x in QUESTIONS if x["id"] in want]
        if not items:
            raise SystemExit("no matching --ids")
    tok_dir = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        trust_remote_code=True,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    rows = []
    for item in items:
        t0 = time.time()
        try:
            text = generate_one(model, tok, item["q"], args.max_new)
            err = None
        except Exception as exc:  # noqa: BLE001
            text, err = "", f"{type(exc).__name__}: {exc}"
        rec = {
            "label": args.label,
            "id": item["id"],
            "question": item["q"],
            "answer": text,
            "n_chars": len(text),
            "md_heading": bool(MD_HEADING.search(text)),
            "loopish": looks_loop(text),
            "has_eos_cut": len(text) >= 20,
            "sec": round(time.time() - t0, 2),
            "error": err,
        }
        rows.append(rec)
        print(json.dumps({"id": item["id"], "n": rec["n_chars"], "loopish": rec["loopish"], "head": text[:180].replace("\n", " / ")}, ensure_ascii=False))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
