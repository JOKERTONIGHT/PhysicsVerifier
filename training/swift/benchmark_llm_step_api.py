#!/usr/bin/env python3
"""Offline DeepSeek judge benchmark and compact-prompt A/B gate.

Uses historical 4x6 GRPO groups. Never reads or sends gold/solution.
API keys are loaded from .env and never printed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.reward_server.llm_step_judge import (  # noqa: E402
    DEFAULT_MODEL,
    LLMStepJudge,
    PROMPT_VERSION_V1,
    PROMPT_VERSION_V2,
    require_remote_model,
)

DEFAULT_COMPLETIONS = Path(
    "/slow_share/jinjianhan/ckpt/qwen3-8b-deepseek-v4-flash-grpo-onset/"
    "v1-20260827-071849/completions.jsonl"
)
REPORT_DIR = ROOT / "logs" / "llm_step_speedup"
NUM_GENERATIONS = 6


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, _, value = raw.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        os.environ[key] = value.strip().strip("'").strip('"')


def extract_question(prompt: str) -> str:
    text = str(prompt or "")
    marker = "<|im_start|>user"
    if marker in text:
        part = text.split(marker, 1)[1]
        part = part.split("<|im_end|>", 1)[0]
        return part.strip()
    return text.strip()


def iter_history_groups(path: Path, *, n_gen: int = NUM_GENERATIONS) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts = obj.get("prompt") or []
            completions = obj.get("completion") or []
            n = min(len(prompts), len(completions))
            for start in range(0, n, n_gen):
                chunk_p = prompts[start : start + n_gen]
                chunk_c = completions[start : start + n_gen]
                if len(chunk_c) != n_gen:
                    continue
                yield {
                    "source_line": line_no,
                    "question": extract_question(chunk_p[0]),
                    "candidates": [str(c or "") for c in chunk_c],
                }


def load_groups(path: Path, limit: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for group in iter_history_groups(path):
        key = group["question"][:240]
        if key in seen:
            continue
        seen.add(key)
        out.append(group)
        if len(out) >= limit:
            break
    return out


def _pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(math.ceil(q * len(ordered)) - 1)))
    return float(ordered[idx])


def _ranks(scores: Sequence[float]) -> List[int]:
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    ranks = [0] * len(scores)
    for rank, idx in enumerate(order):
        ranks[idx] = rank
    return ranks


def _best_id(scores: Sequence[float]) -> int:
    best = 0
    for i, score in enumerate(scores):
        if score > scores[best]:
            best = i
    return best


def _summarize_latencies(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    lats = [float(r["latency_ms"]) for r in rows if r.get("ok")]
    tokens = [float(r.get("completion_tokens") or 0.0) for r in rows if r.get("ok")]
    return {
        "n": float(len(rows)),
        "ok": float(sum(1 for r in rows if r.get("ok"))),
        "schema_ok_rate": (sum(1 for r in rows if r.get("ok")) / max(len(rows), 1)),
        "error_rate": (sum(1 for r in rows if not r.get("ok")) / max(len(rows), 1)),
        "retry_rate": (sum(float(r.get("retries") or 0.0) for r in rows) / max(len(rows), 1)),
        "p50_ms": _pct(lats, 0.50),
        "p95_ms": _pct(lats, 0.95),
        "mean_ms": (sum(lats) / max(len(lats), 1)),
        "mean_completion_tokens": (sum(tokens) / max(len(tokens), 1)),
        "throughput_gps": (len(lats) / max((sum(lats) / 1000.0), 1e-9)) if lats else 0.0,
    }


async def _score_one(
    judge: LLMStepJudge,
    group: Dict[str, Any],
    sem: asyncio.Semaphore,
) -> Dict[str, Any]:
    started = time.time()
    calls_before = judge.calls
    retries_before = judge.retries + judge.parse_retries
    tokens_before = judge.completion_tokens
    async with sem:
        try:
            payloads = await judge.ascore_group(group["question"], group["candidates"])
            ok = True
            err = ""
            scores = [float(p["score"]) for p in payloads]
            fatals = [bool(p["fatal_error"]) for p in payloads]
            answers = [bool(p["answer_only"]) for p in payloads]
        except Exception as exc:  # noqa: BLE001
            ok = False
            err = type(exc).__name__
            payloads = []
            scores = []
            fatals = []
            answers = []
    return {
        "ok": ok,
        "error": err,
        "latency_ms": (time.time() - started) * 1000.0,
        "retries": float((judge.retries + judge.parse_retries) - retries_before),
        "calls": float(judge.calls - calls_before),
        "completion_tokens": float(judge.completion_tokens - tokens_before),
        "scores": scores,
        "fatal_error": fatals,
        "answer_only": answers,
        "zero_std": bool(ok and len(scores) > 1 and statistics.pstdev(scores) <= 1e-12),
        "request_id": getattr(judge, "last_request_id", ""),
        "n_candidates": len(group["candidates"]),
    }


async def run_pool(
    groups: Sequence[Dict[str, Any]],
    *,
    concurrency: int,
    max_tokens: int,
    prompt_version: str,
    max_retries: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    os.environ["LLM_STEP_JUDGE_MAX_TOKENS"] = str(max_tokens)
    os.environ["LLM_STEP_PROMPT_VERSION"] = prompt_version
    os.environ["LLM_STEP_JUDGE_CONCURRENCY"] = str(concurrency)
    judge = LLMStepJudge.from_env()
    judge.max_retries = max_retries
    judge.prompt_version = prompt_version
    judge.max_tokens = max_tokens
    sem = asyncio.Semaphore(max(1, concurrency))
    wall_started = time.time()
    rows = await asyncio.gather(*[_score_one(judge, group, sem) for group in groups])
    wall_s = max(time.time() - wall_started, 1e-9)
    summary = _summarize_latencies(list(rows))
    ok_n = sum(1 for r in rows if r.get("ok"))
    summary["wall_s"] = float(wall_s)
    summary["wall_throughput_gps"] = float(ok_n) / wall_s
    snap = judge.metrics_snapshot()
    summary["judge_inflight_max"] = float(snap.get("judge_inflight_max") or 0.0)
    summary["prompt_tokens"] = float(snap.get("llm_step_prompt_tokens") or 0.0)
    summary["completion_tokens"] = float(snap.get("llm_step_completion_tokens") or 0.0)
    return list(rows), summary


def pick_concurrency(rows: Sequence[Dict[str, Any]]) -> int:
    if not rows:
        return 4
    chosen = rows[0]
    for prev, cur in zip(rows, rows[1:]):
        if float(cur.get("error_rate") or 0.0) > 0.01 or float(cur.get("retry_rate") or 0.0) > 0.01:
            return int(prev["concurrency"])
        base = max(float(prev.get("wall_throughput_gps") or 0.0), 1e-9)
        gain = (float(cur.get("wall_throughput_gps") or 0.0) - base) / base
        if gain < 0.10:
            return int(prev["concurrency"])
        chosen = cur
    return int(chosen["concurrency"])


def pick_max_tokens(rows: Sequence[Dict[str, Any]]) -> int:
    viable = [r for r in rows if float(r.get("schema_ok_rate") or 0.0) >= 0.99]
    if not viable:
        return 4096
    return int(min(viable, key=lambda r: int(r["max_tokens"]))["max_tokens"])


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] wrote {path}")


async def cmd_bench(args: argparse.Namespace) -> Dict[str, Any]:
    groups = load_groups(args.completions, args.groups)
    if len(groups) < 4:
        raise SystemExit(f"need at least 4 historical groups, got {len(groups)}")
    conc_rows: List[Dict[str, Any]] = []
    for conc in args.concurrency:
        _rows, summary = await run_pool(
            groups,
            concurrency=int(conc),
            max_tokens=4096,
            prompt_version=PROMPT_VERSION_V1,
            max_retries=args.max_retries,
        )
        rec = {"concurrency": int(conc), "max_tokens": 4096, "prompt_version": PROMPT_VERSION_V1, **summary}
        conc_rows.append(rec)
        print(json.dumps({"event": "concurrency_point", **rec}, ensure_ascii=False), flush=True)
    token_rows: List[Dict[str, Any]] = []
    for tokens in args.max_tokens:
        _rows, summary = await run_pool(
            groups,
            concurrency=4,
            max_tokens=int(tokens),
            prompt_version=PROMPT_VERSION_V1,
            max_retries=args.max_retries,
        )
        rec = {"concurrency": 4, "max_tokens": int(tokens), "prompt_version": PROMPT_VERSION_V1, **summary}
        token_rows.append(rec)
        print(json.dumps({"event": "token_point", **rec}, ensure_ascii=False), flush=True)
    chosen_conc = pick_concurrency(conc_rows)
    chosen_tokens = pick_max_tokens(token_rows)
    report = {
        "ok": True,
        "n_groups": len(groups),
        "concurrency": conc_rows,
        "max_tokens": token_rows,
        "chosen_concurrency": chosen_conc,
        "chosen_max_tokens": chosen_tokens,
        "effective_train_groups": 4,
        "note": "training still issues at most 4 simultaneous groups; semaphore headroom is for retries",
    }
    write_json(args.out, report)
    return report


def ab_gate(v1: Sequence[Dict[str, Any]], v2: Sequence[Dict[str, Any]], v1_sum: Dict[str, float], v2_sum: Dict[str, float]) -> Dict[str, Any]:
    paired = [(a, b) for a, b in zip(v1, v2) if a.get("ok") and b.get("ok") and a.get("scores") and b.get("scores")]
    n = len(paired)
    rank_ok = 0
    best_ok = 0
    mae = 0.0
    fatal_ok = 0
    answer_ok = 0
    for a, b in paired:
        if _ranks(a["scores"]) == _ranks(b["scores"]):
            rank_ok += 1
        if _best_id(a["scores"]) == _best_id(b["scores"]):
            best_ok += 1
        mae += sum(abs(x - y) for x, y in zip(a["scores"], b["scores"])) / max(len(a["scores"]), 1)
        fatal_ok += int(a.get("fatal_error") == b.get("fatal_error"))
        answer_ok += int(a.get("answer_only") == b.get("answer_only"))
    v1_zero = sum(1 for r in v1 if r.get("ok") and r.get("zero_std")) / max(sum(1 for r in v1 if r.get("ok")), 1)
    v2_zero = sum(1 for r in v2 if r.get("ok") and r.get("zero_std")) / max(sum(1 for r in v2 if r.get("ok")), 1)
    p50_drop = 1.0 - (float(v2_sum.get("p50_ms") or 0.0) / max(float(v1_sum.get("p50_ms") or 1.0), 1e-9))
    p95_drop = 1.0 - (float(v2_sum.get("p95_ms") or 0.0) / max(float(v1_sum.get("p95_ms") or 1.0), 1e-9))
    metrics = {
        "n_scored": n,
        "rank_agree": rank_ok / max(n, 1),
        "best_id_agree": best_ok / max(n, 1),
        "score_mae": mae / max(n, 1),
        "fatal_agree": fatal_ok / max(n, 1),
        "answer_only_agree": answer_ok / max(n, 1),
        "v1_schema_ok_rate": float(v1_sum.get("schema_ok_rate") or 0.0),
        "v2_schema_ok_rate": float(v2_sum.get("schema_ok_rate") or 0.0),
        "v1_zero_std_rate": v1_zero,
        "v2_zero_std_rate": v2_zero,
        "p50_drop": p50_drop,
        "p95_drop": p95_drop,
        "v1_p50_ms": float(v1_sum.get("p50_ms") or 0.0),
        "v1_p95_ms": float(v1_sum.get("p95_ms") or 0.0),
        "v2_p50_ms": float(v2_sum.get("p50_ms") or 0.0),
        "v2_p95_ms": float(v2_sum.get("p95_ms") or 0.0),
    }
    reasons: List[str] = []
    if n < 100:
        reasons.append("n_scored<100")
    if metrics["rank_agree"] < 0.90:
        reasons.append("rank_agree<0.90")
    if metrics["best_id_agree"] < 0.90:
        reasons.append("best_id_agree<0.90")
    if metrics["score_mae"] > 0.05:
        reasons.append("score_mae>0.05")
    if metrics["fatal_agree"] < 0.95:
        reasons.append("fatal_agree<0.95")
    if metrics["answer_only_agree"] < 0.95:
        reasons.append("answer_only_agree<0.95")
    if metrics["v2_schema_ok_rate"] < 0.99:
        reasons.append("v2_schema<0.99")
    if metrics["v2_zero_std_rate"] > metrics["v1_zero_std_rate"] + 0.03:
        reasons.append("zero_std_up>3pp")
    if metrics["p50_drop"] < 0.30 or metrics["p95_drop"] < 0.30:
        reasons.append("latency_drop<30pct")
    return {"ok": not reasons, "reasons": reasons, **metrics}


async def cmd_ab(args: argparse.Namespace) -> Dict[str, Any]:
    groups = load_groups(args.completions, max(args.groups, 100))
    if len(groups) < 100:
        raise SystemExit(f"A/B gate needs >=100 groups, got {len(groups)}")
    v1_rows, v1_sum = await run_pool(
        groups,
        concurrency=4,
        max_tokens=args.v1_max_tokens,
        prompt_version=PROMPT_VERSION_V1,
        max_retries=args.max_retries,
    )
    v2_rows, v2_sum = await run_pool(
        groups,
        concurrency=4,
        max_tokens=args.v2_max_tokens,
        prompt_version=PROMPT_VERSION_V2,
        max_retries=args.max_retries,
    )
    gate = ab_gate(v1_rows, v2_rows, v1_sum, v2_sum)
    report = {
        "n_groups": len(groups),
        "v1": {"prompt_version": PROMPT_VERSION_V1, "max_tokens": args.v1_max_tokens, **v1_sum},
        "v2": {"prompt_version": PROMPT_VERSION_V2, "max_tokens": args.v2_max_tokens, **v2_sum},
        "gate": gate,
        "use_compact": bool(gate.get("ok")),
        "selected_prompt_version": PROMPT_VERSION_V2 if gate.get("ok") else PROMPT_VERSION_V1,
        "selected_max_tokens": args.v2_max_tokens if gate.get("ok") else args.v1_max_tokens,
    }
    write_json(args.out, report)
    print(json.dumps({"event": "ab_gate", "ok": gate.get("ok"), "reasons": gate.get("reasons")}, ensure_ascii=False))
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--completions", type=Path, default=DEFAULT_COMPLETIONS)
    p.add_argument("--env-file", type=Path, default=ROOT / ".env")
    p.add_argument("--max-retries", type=int, default=2)
    sub = p.add_subparsers(dest="cmd", required=True)
    bench = sub.add_parser("bench")
    bench.add_argument("--groups", type=int, default=32)
    bench.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    bench.add_argument("--max-tokens", type=int, nargs="+", default=[4096, 2048, 1536, 1024])
    bench.add_argument("--out", type=Path, default=REPORT_DIR / "api_bench.json")
    ab = sub.add_parser("ab")
    ab.add_argument("--groups", type=int, default=100)
    ab.add_argument("--v1-max-tokens", type=int, default=4096)
    ab.add_argument("--v2-max-tokens", type=int, default=1536)
    ab.add_argument("--out", type=Path, default=REPORT_DIR / "ab_compact.json")
    return p


def main() -> int:
    args = build_parser().parse_args()
    load_env(args.env_file)
    model = os.environ.get("PHYSICSVERIFIER_LLM_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    require_remote_model(model)
    if model != DEFAULT_MODEL:
        raise SystemExit(f"refusing model fallback: {model}")
    if args.cmd == "bench":
        asyncio.run(cmd_bench(args))
        return 0
    if args.cmd == "ab":
        asyncio.run(cmd_ab(args))
        return 0
    raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
