#!/usr/bin/env python3
"""Generate SFT solutions via local 30B (or API fallback) with answer rejection sampling.

Local pass: K samples per prompt, keep the shortest boxed answer that grades as correct.
Remaining misses are written to --unsolved for an optional API fill pass.
Held-out sample_ids are never used.

--hint-gold injects the reference final answer into the *generation* prompt only.
Stored SFT user turns never include that hint, so the student model cannot copy it.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openai import OpenAI

from training.compat.math_grading import grade_answer_verl
from training.rl_data.audit_eval_leakage import load_exclusion, row_excluded
from training.rl_data.screen_training_data import prompt_drop_reason

SFT_SYSTEM = (
    "You are an expert physics competition solver. "
    "Write a complete, rigorous step-by-step solution. "
    "Put the final answer in \\boxed{}."
)
HINT_MARK = "Internal target (do not mention this note, a target, or that any answer was given)"
META_TALK_RE = re.compile(
    r"reference answer|given answer|the given reference|following the given|internal target",
    re.IGNORECASE,
)
DEFAULT_FEWSHOT = Path(__file__).resolve().parent / "sft_fewshot.json"
RETRY_ERRS = {
    "APITimeoutError",
    "APIConnectionError",
    "RateLimitError",
    "InternalServerError",
    "TimeoutError",
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _heldout_ids(path: Optional[Path]) -> set[str]:
    if path is None or not path.is_file():
        return set()
    ids: set[str] = set()
    for row in _load_jsonl(path):
        sid = str(row.get("sample_id") or (row.get("metadata") or {}).get("sample_id") or "")
        if sid:
            ids.add(sid)
    return ids


def _user_question(row: Dict[str, Any]) -> str:
    q = row.get("question")
    if q:
        return str(q)
    for msg in reversed(row.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return ""


def _done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(r.get("sample_id") or "") for r in _load_jsonl(path) if r.get("sample_id")}


def load_fewshot(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None or not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    out: List[Dict[str, str]] = []
    for item in raw:
        q = str(item.get("question") or "").strip()
        sol = str(item.get("solution") or "").strip()
        if q and sol and not META_TALK_RE.search(sol):
            out.append({"question": q, "solution": sol})
    return out


def has_meta_talk(text: str) -> bool:
    return bool(META_TALK_RE.search(text or ""))


def build_generation_messages(
    question: str,
    gold: str,
    hint_gold: bool,
    fewshot: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """Prompt used only while sampling. Never written to the SFT jsonl user turn."""
    system = SFT_SYSTEM
    if fewshot:
        system += (
            " Match the style of the worked examples: derive from the problem, "
            "one \\boxed{} at the end, and never mention a target, hint, or provided answer."
        )
    if hint_gold and gold.strip():
        system += (
            " An internal numerical/symbolic target is attached to the user message "
            "only to keep the final boxed result correct. Write as if solving from scratch."
        )
        user = (
            f"{question}\n\n"
            f"{HINT_MARK}:\n{gold.strip()}\n\n"
            "Derive this result from the problem. Put the final answer in \\boxed{}. "
            "Do not mention the internal target."
        )
    else:
        user = question
    messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
    for ex in fewshot or []:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": ex["solution"]})
    messages.append({"role": "user", "content": user})
    return messages


def user_turn_has_hint(messages: List[Dict[str, Any]]) -> bool:
    for msg in messages or []:
        if str(msg.get("role") or "") != "user":
            continue
        if HINT_MARK in str(msg.get("content") or ""):
            return True
    return False


def is_acceptable_solution(text: str, gold: str, min_chars: int = 0) -> bool:
    if not (text or "").strip():
        return False
    if "\\boxed" not in text:
        return False
    if has_meta_talk(text):
        return False
    if min_chars and len(text) < min_chars:
        return False
    if gold and not grade_answer_verl(text, gold):
        return False
    return True


def _chat(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    retries: int = 3,
) -> str:
    last: Optional[BaseException] = None
    for attempt in range(max(1, retries)):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last = exc
            if type(exc).__name__ not in RETRY_ERRS:
                raise
            time.sleep(min(2 ** attempt, 16))
    raise RuntimeError(f"chat failed after {retries} tries: {last}") from last


def pick_shortest_correct(candidates: List[str], gold: str) -> Optional[str]:
    correct = [c for c in candidates if c.strip() and grade_answer_verl(c, gold)]
    if not correct:
        return None
    return min(correct, key=lambda x: (len(x), x))


def pick_best_correct(candidates: List[str], gold: str, min_chars: int = 0) -> Optional[str]:
    acceptable = [c for c in candidates if is_acceptable_solution(c, gold, min_chars)]
    if acceptable:
        return min(acceptable, key=lambda x: (len(x), x))
    if min_chars:
        graded = [c for c in candidates if is_acceptable_solution(c, gold, 0)]
        if graded:
            return max(graded, key=len)
    return None


def _make_sft_row(src: Dict[str, Any], solution_text: str, *, hint_gold: bool = False) -> Dict[str, Any]:
    messages = list(src.get("messages") or [])
    if not messages:
        messages = [
            {"role": "system", "content": SFT_SYSTEM},
            {"role": "user", "content": _user_question(src)},
        ]
    out_messages = [m for m in messages if m.get("role") != "assistant"]
    if user_turn_has_hint(out_messages):
        raise ValueError("refusing to store gold-hint text in SFT user turns")
    out_messages.append({"role": "assistant", "content": solution_text})
    return {
        "messages": out_messages,
        "solution": src.get("solution") or "",
        "question": src.get("question") or _user_question(src),
        "sample_id": src.get("sample_id"),
        "source": src.get("source"),
        "generator": src.get("generator", "local"),
        "hint_gold": bool(hint_gold),
    }


def generate_for_row(
    client: OpenAI,
    model: str,
    row: Dict[str, Any],
    k: int,
    temperature: float,
    max_tokens: int,
    timeout: float,
    *,
    hint_gold: bool = False,
    min_chars: int = 0,
    fewshot: Optional[Sequence[Dict[str, str]]] = None,
    stop: Optional[threading.Event] = None,
) -> Optional[str]:
    question = _user_question(row)
    gold = str(row.get("solution") or "")
    messages = build_generation_messages(question, gold, hint_gold, fewshot=fewshot)
    cands: List[str] = []
    sid = row.get("sample_id")
    for i in range(k):
        if stop is not None and stop.is_set():
            break
        try:
            print(f"[sft-gen] sample {sid} try {i+1}/{k} hint_gold={int(hint_gold)}", flush=True)
            text = _chat(client, model, messages, temperature, max_tokens, timeout)
            cands.append(text)
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            print(f"[sft-gen] sample {sid} failed try {i+1}: {name}", flush=True)
            if name in {"AuthenticationError", "PermissionDeniedError"}:
                if stop is not None:
                    stop.set()
                print("[sft-gen] API quota/auth failed; stopping fill", flush=True)
                break
            continue
        if is_acceptable_solution(text, gold, min_chars):
            return text
    return pick_best_correct(cands, gold, min_chars)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", type=Path, default=ROOT / "data/rl/swift_prompts.jsonl")
    p.add_argument("--heldout", type=Path, default=ROOT / "data/rl/heldout_eval.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/rl/sft_solutions.jsonl")
    p.add_argument("--unsolved", type=Path, default=ROOT / "data/rl/sft_unsolved.jsonl")
    p.add_argument("--report", type=Path, default=ROOT / "data/rl/sft_gen_report.json")
    p.add_argument("--base-url", default=os.environ.get("SFT_GEN_BASE_URL", "http://127.0.0.1:8780/v1"))
    p.add_argument("--model", default=os.environ.get("SFT_GEN_MODEL", "qwen3-30b-a3b"))
    p.add_argument("--api-base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument(
        "--api-model",
        default=os.environ.get("SFT_API_MODEL", "deepseek-v4-flash"),
    )
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--api-k", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--max-prompts", type=int, default=0)
    p.add_argument("--target-solved", type=int, default=0, help="Stop once this many correct SFT rows exist (0=all)")
    p.add_argument("--local-only", action="store_true")
    p.add_argument("--api-only", action="store_true")
    p.add_argument(
        "--hint-gold",
        action=argparse.BooleanOptionalAction,
        default=_env_bool("SFT_HINT_GOLD", False),
        help="Show gold answer in the generation prompt only (not stored in SFT messages).",
    )
    p.add_argument("--min-chars", type=int, default=int(os.environ.get("SFT_MIN_CHARS", "400")))
    p.add_argument(
        "--fewshot",
        type=Path,
        default=Path(os.environ.get("SFT_FEWSHOT", str(DEFAULT_FEWSHOT))),
        help="JSON list of {question, solution} exemplars for the generation prompt only.",
    )
    args = p.parse_args()
    fewshot = load_fewshot(args.fewshot)

    if not args.prompts.is_file():
        raise SystemExit(f"prompt file missing: {args.prompts}")

    heldout = _heldout_ids(args.heldout)
    excl_ids, excl_hashes = load_exclusion(ROOT / "data/rl/train_manifest.json")
    rows = []
    for r in _load_jsonl(args.prompts):
        sid = str(r.get("sample_id") or "")
        if sid in heldout:
            continue
        if row_excluded(r, excl_ids, excl_hashes):
            continue
        if prompt_drop_reason(r):
            continue
        rows.append(r)
    if args.max_prompts:
        rows = rows[: args.max_prompts]
    done = _done_ids(args.output)
    todo = [r for r in rows if str(r.get("sample_id") or "") not in done]
    if args.target_solved:
        need = max(args.target_solved - len(done), 0)
        if need:
            todo = todo[: max(need * 4, need + args.concurrency * 8)]
    args.output.parent.mkdir(parents=True, exist_ok=True)

    solved = len(done)
    failed: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    stop = threading.Event()
    local_client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    api_client = None
    if args.api_base_url and args.api_key and args.api_model and not args.local_only:
        api_client = OpenAI(base_url=args.api_base_url, api_key=args.api_key, timeout=args.timeout)
    if args.api_only and api_client is None:
        raise SystemExit("API fill requires OPENAI_BASE_URL, OPENAI_API_KEY, and a model name")
    if args.target_solved and solved >= args.target_solved:
        print(json.dumps({"n_solved": solved, "already_at_target": True}, indent=2))
        return 0

    min_chars = args.min_chars if args.hint_gold else 0

    def work(row: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[str], str]:
        if stop.is_set():
            return row, None, "stopped"
        stage = "skip"
        text = None
        if not args.api_only:
            stage = "local"
            text = generate_for_row(
                local_client,
                args.model,
                row,
                args.k,
                args.temperature,
                args.max_tokens,
                args.timeout,
                hint_gold=args.hint_gold,
                min_chars=min_chars,
                fewshot=fewshot,
                stop=stop,
            )
        if text is None and api_client is not None and not stop.is_set():
            stage = "api"
            api_k = args.k if args.api_only else args.api_k
            text = generate_for_row(
                api_client,
                args.api_model,
                row,
                api_k,
                args.temperature,
                args.max_tokens,
                args.timeout,
                hint_gold=args.hint_gold,
                min_chars=min_chars,
                fewshot=fewshot,
                stop=stop,
            )
        return row, text, stage

    with args.output.open("a", encoding="utf-8") as fout:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futs = [pool.submit(work, row) for row in todo]
            for i, fut in enumerate(as_completed(futs), 1):
                row, text, stage = fut.result()
                if stage == "stopped":
                    skipped.append(row)
                    continue
                if text is None:
                    failed.append(row)
                else:
                    rec = _make_sft_row(row, text, hint_gold=args.hint_gold)
                    rec["generator"] = stage
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    solved += 1
                    if args.target_solved and solved >= args.target_solved:
                        stop.set()
                if i % 20 == 0:
                    print(f"[sft-gen] done {i}/{len(todo)} solved_total={solved}", flush=True)

    args.unsolved.parent.mkdir(parents=True, exist_ok=True)
    with args.unsolved.open("w", encoding="utf-8") as uf:
        for row in failed:
            uf.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "n_prompts": len(rows),
        "n_heldout_excluded": len(heldout),
        "n_already_done": len(done),
        "n_solved": solved,
        "n_unsolved": len(failed),
        "n_stopped_early": len(skipped),
        "coverage": solved / max(len(rows), 1),
        "output": str(args.output),
        "unsolved": str(args.unsolved),
        "target_solved": args.target_solved,
        "hint_gold": bool(args.hint_gold),
        "n_fewshot": len(fewshot),
        "api_model": args.api_model if args.api_only or api_client is not None else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
