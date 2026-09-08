#!/usr/bin/env python3
"""PhysicsVerifier reward server for OpenRLHF and legacy HTTP clients.

Endpoints:
  POST /            - score single sample (slime)
  POST /batch       - score batch of samples (slime)
  POST /get_reward  - OpenRLHF remote RM ({query, prompts, labels} -> rewards)

Payload (single /):
  {prompt, response, label, question, ...}

Response (/):
  {score, acc, n_errors, diagnostics_summary, ...}

OpenRLHF (/get_reward):
  Request:  {query: [prompt+response...], prompts: [...], labels: [...]}
  Response: {rewards: [...], scores: [...], extra_logs: {...}}
"""
from __future__ import annotations

import argparse
import asyncio
import ast
import copy
import hashlib
import json
import os
import sys
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Project root on PYTHONPATH
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.physics_rule_verifier import PhysicsRuleVerifier
from training.compat.math_grading import extract_answer, grade_answer_verl
from training.reward_server.llm_step_judge import (
    DEFAULT_MODEL as LLM_STEP_MODEL,
    LLMStepJudge,
    LLMStepJudgeError,
    PROMPT_VERSION as LLM_STEP_PROMPT_VERSION,
    group_cache_key as llm_step_group_cache_key,
)
from training.reward_server.paragraph_process import (
    ProcessParagraphWeights,
    score_text_with_diagnostics,
)

app = FastAPI(title="PhysicsVerifier Reward Server")

_verifier: Optional[PhysicsRuleVerifier] = None
_semaphore: Optional[asyncio.Semaphore] = None
_verifier_executor: Optional[ThreadPoolExecutor] = None
_llm_judge: Optional[LLMStepJudge] = None

DEFAULT_LAMBDA = float(os.environ.get("PHYSICS_REWARD_LAMBDA", "0.3"))
DEFAULT_CAP = int(os.environ.get("PHYSICS_REWARD_ERROR_CAP", "3"))
DEFAULT_CONCURRENCY = int(os.environ.get("PHYSICS_REWARD_CONCURRENCY", "4"))
REWARD_MODE = os.environ.get("PHYSICS_REWARD_MODE", "answer_low_verifier").strip().lower()
W_ANSWER = float(os.environ.get("PHYSICS_REWARD_W_ANSWER", "1.0"))
W_FORMAT = float(os.environ.get("PHYSICS_REWARD_W_FORMAT", "0.05"))
W_VERIFIER = float(os.environ.get("PHYSICS_REWARD_W_VERIFIER", "0.1"))
W_LENGTH = float(os.environ.get("PHYSICS_REWARD_W_LENGTH", "0.0"))
W_PROCESS = float(os.environ.get("PHYSICS_REWARD_W_PROCESS", "0.3"))
PROCESS_ALPHA = float(os.environ.get("PHYSICS_REWARD_PROCESS_ALPHA", "0.2"))
VERIFIER_SAMPLE_RATE = float(os.environ.get("PHYSICS_VERIFIER_SAMPLE_RATE", "1.0"))
VERIFIER_ON_WRONG = os.environ.get("PHYSICS_REWARD_VERIFIER_ON_WRONG", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MAX_RESPONSE_CHARS = int(os.environ.get("PHYSICS_REWARD_MAX_RESPONSE_CHARS", "12000"))
PARA_MIN_LEN = int(os.environ.get("PHYSICS_REWARD_PARA_MIN", "150") or 150)
PARA_TARGET_LEN = int(os.environ.get("PHYSICS_REWARD_PARA_TARGET", "220") or 220)
PARA_MAX_LEN = int(os.environ.get("PHYSICS_REWARD_PARA_MAX", "280") or 280)
W_CLEAN = float(os.environ.get("PHYSICS_REWARD_W_CLEAN", "0.5"))
W_FIRST = float(os.environ.get("PHYSICS_REWARD_W_FIRST", "0.3"))
W_DENSE = float(os.environ.get("PHYSICS_REWARD_W_DENSE", "0.2"))
METRICS_LOG = os.environ.get("PHYSICS_REWARD_METRICS_LOG", str(ROOT / "logs/physics_reward_metrics.jsonl"))
REWARD_CACHE_SIZE = int(os.environ.get("PHYSICS_REWARD_CACHE_SIZE", "4096") or 0)
DEFAULT_UNIFIED_RULES = ROOT / "catalogs/rules_unified_3000_runtime_backfilled.json"
SYMBOLIC_ENABLED = os.environ.get("PHYSICSVERIFIER_SYMBOLIC_ENABLED", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def reward_cache_key(question: str, response: str, label: Any = None, *, include_label: bool = True) -> str:
    labels = _normalize_label(label) if include_label else []
    payload = "\0".join([str(question or ""), str(response or ""), *labels]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def group_indices_by_key(keys: Sequence[str]) -> List[List[int]]:
    """Group indices that share a key, preserving first-seen key order."""
    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for idx, key in enumerate(keys):
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(idx)
    return [groups[key] for key in order]


class RewardResultCache:
    """Process-local LRU for (question, response) -> score payload."""

    def __init__(self, maxsize: int = 0) -> None:
        self.maxsize = max(0, int(maxsize))
        self._data: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if self.maxsize <= 0:
            return None
        with self._lock:
            cached = self._data.get(key)
            if cached is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(cached)

    def put(self, key: str, value: Dict[str, Any]) -> None:
        if self.maxsize <= 0 or not isinstance(value, dict):
            return
        with self._lock:
            self._data[key] = copy.deepcopy(value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)

    def reset(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, float]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": float(len(self._data)),
                "hits": float(self.hits),
                "misses": float(self.misses),
                "hit_rate": float(self.hits) / float(total) if total else 0.0,
            }


_reward_cache = RewardResultCache(REWARD_CACHE_SIZE)


def reset_reward_cache(maxsize: Optional[int] = None) -> None:
    global _reward_cache, REWARD_CACHE_SIZE
    if maxsize is not None:
        REWARD_CACHE_SIZE = max(0, int(maxsize))
    _reward_cache = RewardResultCache(REWARD_CACHE_SIZE)


class ScoreRequest(BaseModel):
    prompt: str = ""
    response: str = ""
    label: Optional[List[str] | str] = None
    question: Optional[str] = None
    points: Optional[List[float]] = None
    marking: Any = None
    marking_mode: str = "total_score"
    use_xverify: bool = False


class BatchRequest(BaseModel):
    requests: List[ScoreRequest]


def _normalize_label(label: Optional[List[str] | str]) -> List[str]:
    if label is None:
        return []
    if isinstance(label, list):
        return [str(x) for x in label if x is not None]
    text = str(label).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return [text]


def _extract_question(req: ScoreRequest) -> str:
    if req.question and req.question.strip():
        return req.question.strip()
    prompt = req.prompt
    if isinstance(prompt, list):
        parts = []
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("content"):
                parts.append(str(msg["content"]))
        return "\n".join(parts).strip()
    return str(prompt).strip()


def _check_answer(response: str, labels: List[str]) -> tuple[bool, str, str]:
    if not labels:
        return False, "", ""
    extracted_pred = extract_answer(response) or ""
    for gt in labels:
        if grade_answer_verl(response, gt):
            extracted_gt = extract_answer(gt) or str(gt)
            return True, extracted_pred, extracted_gt
    gt0 = labels[0]
    extracted_gt = extract_answer(gt0) or str(gt0)
    return False, extracted_pred, extracted_gt


def _count_error_diagnostics(result: Dict[str, Any]) -> tuple[int, List[Dict[str, str]]]:
    diagnostics = result.get("diagnostics") or []
    errors = [d for d in diagnostics if str(d.get("severity", "")).lower() == "error"]
    summary = []
    for d in errors[:5]:
        summary.append(
            {
                "rule": str(d.get("rule", "")),
                "message": str(d.get("message", ""))[:200],
            }
        )
    return len(errors), summary


def _compute_score(acc: bool, n_errors: int, *, lam: float, cap: int) -> float:
    penalty = min(n_errors, cap) / max(cap, 1)
    return (1.0 if acc else 0.0) - lam * penalty


def _process_paragraph_mode() -> bool:
    return REWARD_MODE == "process_paragraph"


def _llm_step_mode() -> bool:
    return REWARD_MODE == "llm_step_score"


def _hybrid_llm_outcome_mode() -> bool:
    return REWARD_MODE == "hybrid_llm_outcome"


def _outcome_only_mode() -> bool:
    return REWARD_MODE == "outcome_only"


def _uses_llm_judge() -> bool:
    return _llm_step_mode() or _hybrid_llm_outcome_mode()


def combine_hybrid_llm_outcome(
    *,
    acc: bool,
    boxed: bool,
    process: float,
    w_answer: float = 1.0,
    w_format: float = 0.05,
    w_process: float = 0.3,
    alpha: float = 0.2,
) -> float:
    """Outcome-gated mix: any correct score must beat any incorrect score.

    R = w_answer * acc + w_format * boxed + w_process * process * (alpha + (1-alpha) * acc)
    """
    acc_f = 1.0 if acc else 0.0
    boxed_f = 1.0 if boxed else 0.0
    proc = max(0.0, min(1.0, float(process)))
    alpha_f = max(0.0, min(1.0, float(alpha)))
    gate = alpha_f + (1.0 - alpha_f) * acc_f
    return float(w_answer) * acc_f + float(w_format) * boxed_f + float(w_process) * proc * gate


def _label_text(label: Any) -> str:
    if label is None:
        return ""
    if isinstance(label, (list, tuple)):
        return " ".join(str(x) for x in label if x is not None).strip()
    return str(label).strip()


def hybrid_labels_missing(labels: Sequence[Any] | None, n: int) -> bool:
    if n <= 0:
        return False
    if not labels:
        return True
    padded = list(labels)
    if len(padded) < n:
        return True
    return any(not _label_text(padded[i]) for i in range(n))


def _process_weights() -> ProcessParagraphWeights:
    # Hard-zero answer/format so GRPO cannot latch onto boxed-answer correctness.
    return ProcessParagraphWeights(
        clean=max(W_CLEAN, 0.0),
        first=max(W_FIRST, 0.0),
        dense=max(W_DENSE, 0.0),
        answer=0.0,
        format=0.0,
    )


def _reward_weights() -> Dict[str, float]:
    mode = REWARD_MODE
    if mode == "llm_step_score":
        return {"answer": 0.0, "format": 0.0, "verifier": 0.0, "length": 0.0, "llm_step": 1.0}
    if mode == "hybrid_llm_outcome":
        return {
            "answer": max(W_ANSWER, 0.0),
            "format": max(W_FORMAT, 0.0),
            "verifier": 0.0,
            "length": 0.0,
            "llm_step": max(W_PROCESS, 0.0),
            "process_alpha": max(0.0, min(1.0, PROCESS_ALPHA)),
        }
    if mode == "outcome_only":
        return {
            "answer": max(W_ANSWER, 0.0),
            "format": max(W_FORMAT, 0.0),
            "verifier": 0.0,
            "length": 0.0,
            "llm_step": 0.0,
        }
    if mode == "answer_only":
        return {"answer": 1.0, "format": 0.0, "verifier": 0.0, "length": 0.0}
    if mode == "process_paragraph":
        pw = _process_weights()
        return {
            "answer": pw.answer,
            "format": pw.format,
            "verifier": 0.0,
            "length": max(W_LENGTH, 0.0),
            "clean": pw.clean,
            "first": pw.first,
            "dense": pw.dense,
        }
    if mode == "answer_full_verifier":
        return {
            "answer": max(W_ANSWER, 0.0),
            "format": max(W_FORMAT, 0.0),
            "verifier": max(W_VERIFIER, DEFAULT_LAMBDA),
            "length": max(W_LENGTH, 0.0),
        }
    return {
        "answer": max(W_ANSWER, 0.0),
        "format": max(W_FORMAT, 0.0),
        "verifier": max(W_VERIFIER, 0.1),
        "length": max(W_LENGTH, 0.0),
    }


def _format_component(response: str) -> float:
    extracted = extract_answer(response) or ""
    return 1.0 if extracted else 0.0


def _length_penalty(response: str) -> float:
    if MAX_RESPONSE_CHARS <= 0:
        return 0.0
    over = max(len(response) - MAX_RESPONSE_CHARS, 0)
    return min(over / max(MAX_RESPONSE_CHARS, 1), 1.0)


def _error_diagnostics(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    diagnostics = result.get("diagnostics") or []
    if _process_paragraph_mode():
        allowed = {"error", "warning"}
        return [
            d
            for d in diagnostics
            if isinstance(d, dict) and str(d.get("severity", "")).lower() in allowed
        ]
    return [d for d in diagnostics if isinstance(d, dict) and str(d.get("severity", "")).lower() == "error"]


def _should_run_verifier(acc: bool, sample_idx: int) -> bool:
    if _uses_llm_judge() or _outcome_only_mode():
        return False
    if _process_paragraph_mode() or VERIFIER_ON_WRONG:
        pass
    elif not acc:
        return False
    weights = _reward_weights()
    if (not _process_paragraph_mode()) and weights.get("verifier", 0.0) <= 0.0:
        return False
    if VERIFIER_SAMPLE_RATE >= 1.0:
        return True
    if VERIFIER_SAMPLE_RATE <= 0.0:
        return False
    return (sample_idx % max(int(round(1.0 / VERIFIER_SAMPLE_RATE)), 1)) == 0


def _append_metrics(record: Dict[str, Any]) -> None:
    try:
        path = Path(METRICS_LOG)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


@lru_cache(maxsize=1)
def _get_verifier() -> PhysicsRuleVerifier:
    default_precision = "balanced" if _process_paragraph_mode() else "strict"
    precision_mode = os.environ.get("PHYSICSVERIFIER_PRECISION_MODE", default_precision).strip().lower()
    if precision_mode not in {"strict", "balanced", "score_only"}:
        precision_mode = default_precision
    return PhysicsRuleVerifier(
        llm_model=os.environ.get("PHYSICSVERIFIER_LLM_MODEL", "qwen3-30b-a3b"),
        unified_rules_path=os.environ.get(
            "PHYSICSVERIFIER_UNIFIED_RULES",
            str(DEFAULT_UNIFIED_RULES),
        ),
        experience_code_manifest_path=os.environ.get(
            "PHYSICSVERIFIER_SYMBOLIC_MANIFEST",
            str(ROOT / "results/experience_symbolic_program_manifest_v2_unified.json"),
        ),
        enable_symbolic_check=SYMBOLIC_ENABLED,
        unified_retrieval_mode=os.environ.get(
            "PHYSICSVERIFIER_UNIFIED_RETRIEVAL_MODE",
            "lexical" if _process_paragraph_mode() else "semantic",
        ),
        semantic_output_adapter=os.environ.get("PHYSICSVERIFIER_SEMANTIC_OUTPUT_ADAPTER") or None,
        precision_mode=precision_mode,
    )


async def _invoke_llm_score_group(judge: Any, question: str, solutions: Sequence[str]) -> List[Dict[str, Any]]:
    ascore = getattr(judge, "ascore_group", None)
    if callable(ascore):
        return await ascore(question, solutions)
    return await asyncio.to_thread(judge.score_group, question, solutions)


def _get_llm_step_judge() -> LLMStepJudge:
    global _llm_judge
    if _llm_judge is None:
        _llm_judge = LLMStepJudge.from_env()
    return _llm_judge


def _llm_step_payload(item: Dict[str, Any], *, latency_ms: float, cache_hit: bool) -> Dict[str, Any]:
    score = float(item.get("score") or 0.0)
    return {
        "score": score,
        "point": score,
        "acc": False,
        "n_errors": 0,
        "extracted_pred": "",
        "extracted_gt": "",
        "diagnostics_summary": [],
        "scored_by": "deepseek_llm_step_score",
        "verifier_mode": "skipped",
        "score_noxverify": score,
        "point_noxverify": score,
        "prompt_version": LLM_STEP_PROMPT_VERSION,
        "judge_model": LLM_STEP_MODEL,
        "reward_components": {
            "answer": 0.0,
            "format": 0.0,
            "verifier": 0.0,
            "length": 0.0,
            "llm_step": score,
            "raw_score": item.get("raw_score"),
            "fatal_error": bool(item.get("fatal_error")),
            "answer_only": bool(item.get("answer_only")),
            "brief_reason": item.get("brief_reason"),
            "step_assessments": item.get("step_assessments") or [],
            "weights": _reward_weights(),
            "verifier_failed": False,
        },
        "latency_ms": latency_ms,
        "cache_hit": cache_hit,
    }


def apply_hybrid_outcome(payload: Dict[str, Any], response: str, label: Any) -> Dict[str, Any]:
    """Fold local acc/boxed onto a DeepSeek process payload. Gold never goes to the judge."""
    labels = _normalize_label(label)
    if not labels:
        raise HTTPException(status_code=400, detail="hybrid_llm_outcome requires labels; refusing process-only fallback")
    acc, extracted_pred, extracted_gt = _check_answer(response, labels)
    boxed = bool(_format_component(response))
    process = float((payload.get("reward_components") or {}).get("llm_step") or payload.get("score") or 0.0)
    weights = _reward_weights()
    score = combine_hybrid_llm_outcome(
        acc=acc,
        boxed=boxed,
        process=process,
        w_answer=float(weights.get("answer") or 0.0),
        w_format=float(weights.get("format") or 0.0),
        w_process=float(weights.get("llm_step") or 0.0),
        alpha=float(weights.get("process_alpha") or 0.0),
    )
    out = copy.deepcopy(payload)
    out["score"] = score
    out["point"] = score
    out["acc"] = acc
    out["extracted_pred"] = extracted_pred
    out["extracted_gt"] = extracted_gt
    out["scored_by"] = "hybrid_llm_outcome"
    comps = dict(out.get("reward_components") or {})
    comps["answer"] = 1.0 if acc else 0.0
    comps["format"] = 1.0 if boxed else 0.0
    comps["llm_step"] = process
    comps["hybrid"] = score
    comps["process_gate"] = float(weights.get("process_alpha") or 0.0) + (
        (1.0 - float(weights.get("process_alpha") or 0.0)) * (1.0 if acc else 0.0)
    )
    comps["weights"] = weights
    out["reward_components"] = comps
    return out


async def score_one(req: ScoreRequest, sample_idx: int = 0) -> Dict[str, Any]:
    global _semaphore, _verifier_executor
    if _semaphore is None or _verifier_executor is None:
        _init_runtime(DEFAULT_CONCURRENCY)

    started = time.time()
    question = _extract_question(req)
    response_text = _effective_response(req.response)
    if _uses_llm_judge():
        if _hybrid_llm_outcome_mode() and hybrid_labels_missing([req.label], 1):
            raise HTTPException(
                status_code=400,
                detail="hybrid_llm_outcome requires labels; refusing process-only fallback",
            )
        judge = _get_llm_step_judge()
        cache_key = llm_step_group_cache_key(question, [response_text], prompt_version=judge.prompt_version)
        cached = _reward_cache.get(cache_key)
        if cached is not None and cached.get("payloads"):
            out = _llm_step_payload(cached["payloads"][0], latency_ms=(time.time() - started) * 1000.0, cache_hit=True)
        else:
            try:
                payloads = await _invoke_llm_score_group(judge, question, [response_text])
            except LLMStepJudgeError as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc
            _reward_cache.put(cache_key, {"payloads": payloads})
            out = _llm_step_payload(payloads[0], latency_ms=(time.time() - started) * 1000.0, cache_hit=False)
        if _hybrid_llm_outcome_mode():
            out = apply_hybrid_outcome(out, response_text, req.label)
        return out

    labels = _normalize_label(req.label)
    cache_key = reward_cache_key(question, response_text, labels)
    cached = _reward_cache.get(cache_key)
    if cached is not None:
        cached["latency_ms"] = (time.time() - started) * 1000.0
        cached["cache_hit"] = True
        return cached
    acc, extracted_pred, extracted_gt = _check_answer(response_text, labels)

    n_errors = 0
    diagnostics_summary: List[Dict[str, str]] = []
    error_diags: List[Dict[str, Any]] = []
    verifier_mode = "skipped"
    verifier_failed = False

    if _should_run_verifier(acc, sample_idx):
        async with _semaphore:
            loop = asyncio.get_event_loop()
            verifier_mode = "full"
            try:
                result = await loop.run_in_executor(
                    _verifier_executor,
                    lambda q=question, pred=response_text: _get_verifier().verify(
                        {"question": q, "prediction": pred}
                    ),
                )
                error_diags = _error_diagnostics(result)
                n_errors, diagnostics_summary = _count_error_diagnostics(result)
            except Exception as exc:
                verifier_failed = True
                verifier_mode = "failed"
                diagnostics_summary = [{"rule": "verifier_error", "message": str(exc)[:200]}]

    weights = _reward_weights()
    r_answer = 1.0 if acc else 0.0
    r_format = _format_component(response_text)
    penalty = min(n_errors, DEFAULT_CAP) / max(DEFAULT_CAP, 1)
    r_verifier = -penalty if verifier_mode == "full" else 0.0
    r_length = -_length_penalty(response_text)
    process_components: Dict[str, Any] = {}
    if _process_paragraph_mode():
        if verifier_failed:
            process_components = {
                "n_paragraphs": 0,
                "n_errors": 0,
                "n_bad_paragraphs": 0,
                "first_bad_paragraph_index": None,
                "r_clean": 0.0,
                "r_first": 0.0,
                "r_dense": 0.0,
                "score": 0.0,
            }
            score = float(process_components["score"]) + weights.get("length", 0.0) * r_length
        else:
            process_components = score_text_with_diagnostics(
                response_text,
                error_diags,
                acc=acc,
                boxed=bool(r_format),
                weights=_process_weights(),
                min_len=PARA_MIN_LEN,
                target_len=PARA_TARGET_LEN,
                max_len=PARA_MAX_LEN,
            )
            score = float(process_components.get("score") or 0.0) + weights.get("length", 0.0) * r_length
    else:
        score = (
            weights["answer"] * r_answer
            + weights["format"] * r_format
            + weights["verifier"] * r_verifier
            + weights["length"] * r_length
        )
    legacy_score = _compute_score(acc, n_errors, lam=DEFAULT_LAMBDA, cap=DEFAULT_CAP)
    latency_ms = (time.time() - started) * 1000.0
    out = {
        "score": score,
        "point": score,
        "acc": acc,
        "n_errors": n_errors,
        "extracted_pred": extracted_pred,
        "extracted_gt": extracted_gt,
        "diagnostics_summary": diagnostics_summary,
        "scored_by": "outcome_only" if _outcome_only_mode() else "physics_verifier",
        "verifier_mode": verifier_mode,
        "score_noxverify": legacy_score,
        "point_noxverify": legacy_score,
        "reward_components": {
            "answer": r_answer,
            "format": r_format,
            "verifier_penalty": penalty,
            "verifier": r_verifier,
            "length": r_length,
            "weights": weights,
            "verifier_failed": verifier_failed,
            "process_paragraph": process_components,
        },
        "n_paragraphs": process_components.get("n_paragraphs"),
        "n_bad_paragraphs": process_components.get("n_bad_paragraphs"),
        "latency_ms": latency_ms,
    }
    _append_metrics(
        {
            "ts": time.time(),
            "acc": acc,
            "score": score,
            "legacy_score": legacy_score,
            "verifier_mode": verifier_mode,
            "n_errors": n_errors,
            "n_paragraphs": process_components.get("n_paragraphs"),
            "n_bad_paragraphs": process_components.get("n_bad_paragraphs"),
            "r_clean": process_components.get("r_clean"),
            "r_first": process_components.get("r_first"),
            "r_dense": process_components.get("r_dense"),
            "latency_ms": latency_ms,
            "reward_mode": REWARD_MODE,
        }
    )
    out["cache_hit"] = False
    _reward_cache.put(cache_key, out)
    return out


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/")
async def score_endpoint(req: ScoreRequest) -> Dict[str, Any]:
    return await score_one(req)


@app.post("/batch")
async def score_batch(req: BatchRequest) -> Dict[str, List[Dict[str, Any]]]:
    results = await asyncio.gather(*[score_one(r) for r in req.requests])
    return {"results": list(results)}


class OpenRLHFRewardRequest(BaseModel):
    """OpenRLHF remote RM payload (see openrlhf.utils.remote_rm_utils)."""

    query: List[str] = Field(default_factory=list)
    prompts: List[str] = Field(default_factory=list)
    labels: List[Any] = Field(default_factory=list)


def _effective_response(text: str) -> str:
    src = str(text or "")
    if MAX_RESPONSE_CHARS <= 0:
        return src
    return src[:MAX_RESPONSE_CHARS]


def _init_runtime(concurrency: int) -> None:
    global DEFAULT_CONCURRENCY, _semaphore, _verifier_executor
    DEFAULT_CONCURRENCY = max(1, int(concurrency))
    _semaphore = asyncio.Semaphore(DEFAULT_CONCURRENCY)
    if _verifier_executor is not None:
        _verifier_executor.shutdown(wait=False, cancel_futures=True)
    _verifier_executor = ThreadPoolExecutor(max_workers=DEFAULT_CONCURRENCY)


def _response_from_query(query: str, prompt: str) -> str:
    if prompt and isinstance(prompt, str) and prompt in query:
        return query[len(prompt) :]
    return query


def _llm_step_group_logs(
    questions: Sequence[str],
    rewards: Sequence[float],
    payloads: Sequence[Dict[str, Any]],
    *,
    cache_hits: int,
    unique_groups: int,
) -> Dict[str, float]:
    import statistics

    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for idx, q in enumerate(questions):
        if q not in groups:
            order.append(q)
            groups[q] = []
        groups[q].append(idx)
    stds: List[float] = []
    ranges: List[float] = []
    zero_std = 0
    for q in order:
        vals = [float(rewards[i]) for i in groups[q]]
        if len(vals) <= 1:
            stds.append(0.0)
            ranges.append(0.0)
            zero_std += 1
            continue
        std = statistics.pstdev(vals)
        stds.append(std)
        ranges.append(max(vals) - min(vals))
        if std <= 1e-12:
            zero_std += 1
    n = max(len(rewards), 1)
    sat01 = sum(1 for r in rewards if r <= 1e-8 or r >= 1.0 - 1e-8) / n
    fatal = sum(1 for p in payloads if (p.get("reward_components") or {}).get("fatal_error")) / n
    answer_only = sum(1 for p in payloads if (p.get("reward_components") or {}).get("answer_only")) / n
    judge = _get_llm_step_judge()
    snap = judge.metrics_snapshot()
    return {
        "physics_reward_mode": 0.0,
        "physics_llm_step_group_mean": float(sum(rewards) / n),
        "physics_llm_step_group_std_mean": float(sum(stds) / max(len(stds), 1)),
        "physics_llm_step_group_range_mean": float(sum(ranges) / max(len(ranges), 1)),
        "physics_llm_step_zero_std_rate": float(zero_std) / max(len(stds), 1),
        "physics_llm_step_sat01_rate": float(sat01),
        "physics_llm_step_fatal_rate": float(fatal),
        "physics_llm_step_answer_only_rate": float(answer_only),
        "physics_reward_cache_hit_rate": float(cache_hits) / n,
        "physics_reward_batch_cache_hits": float(cache_hits),
        "physics_reward_batch_unique_scored": float(unique_groups),
        **snap,
    }


def _hybrid_group_logs(
    questions: Sequence[str],
    rewards: Sequence[float],
    process_scores: Sequence[float],
    accs: Sequence[float],
    payloads: Sequence[Dict[str, Any]],
    *,
    cache_hits: int,
    unique_groups: int,
) -> Dict[str, float]:
    extra = _llm_step_group_logs(
        questions,
        rewards,
        payloads,
        cache_hits=cache_hits,
        unique_groups=unique_groups,
    )
    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for idx, q in enumerate(questions):
        if q not in groups:
            order.append(q)
            groups[q] = []
        groups[q].append(idx)
    mixed = 0
    hybrid_zero = 0
    for q in order:
        idxs = groups[q]
        acc_vals = [float(accs[i]) for i in idxs]
        rew_vals = [float(rewards[i]) for i in idxs]
        if acc_vals and min(acc_vals) < 0.5 <= max(acc_vals):
            mixed += 1
        if len(rew_vals) <= 1 or (max(rew_vals) - min(rew_vals)) <= 1e-12:
            hybrid_zero += 1
    n = max(len(rewards), 1)
    extra["physics_llm_step_group_mean"] = float(sum(process_scores) / max(len(process_scores), 1))
    extra["physics_hybrid_reward_mean"] = float(sum(rewards) / n)
    extra["physics_answer_acc"] = float(sum(accs) / n)
    extra["physics_acc"] = extra["physics_answer_acc"]
    extra["physics_mixed_acc_group_rate"] = float(mixed) / max(len(order), 1)
    extra["physics_hybrid_zero_std_rate"] = float(hybrid_zero) / max(len(order), 1)
    extra["physics_hybrid_mode"] = 1.0
    extra["physics_format_rate"] = float(
        sum(1.0 for p in payloads if (p.get("reward_components") or {}).get("format")) / n
    )
    return extra


def _outcome_group_logs(
    questions: Sequence[str],
    rewards: Sequence[float],
    accs: Sequence[float],
    payloads: Sequence[Dict[str, Any]],
    *,
    cache_hits: int,
    unique_scored: int,
) -> Dict[str, Any]:
    """Local acc/boxed mix: mixed groups and within-group reward std."""
    import statistics

    groups: Dict[str, List[int]] = {}
    order: List[str] = []
    for idx, q in enumerate(questions):
        if q not in groups:
            order.append(q)
            groups[q] = []
        groups[q].append(idx)
    mixed = 0
    zero_std = 0
    stds: List[float] = []
    for q in order:
        idxs = groups[q]
        acc_vals = [float(accs[i]) for i in idxs]
        rew_vals = [float(rewards[i]) for i in idxs]
        if acc_vals and min(acc_vals) < 0.5 <= max(acc_vals):
            mixed += 1
        if len(rew_vals) <= 1 or (max(rew_vals) - min(rew_vals)) <= 1e-12:
            zero_std += 1
            stds.append(0.0)
        else:
            stds.append(float(statistics.pstdev(rew_vals)))
    n = max(len(rewards), 1)
    n_groups = max(len(order), 1)
    format_rate = 0.0
    if payloads:
        format_rate = float(
            sum(1.0 for p in payloads if (p.get("reward_components") or {}).get("format")) / n
        )
    return {
        "physics_reward_mode": REWARD_MODE,
        "physics_answer_acc": float(sum(accs) / n),
        "physics_acc": float(sum(accs) / n),
        "physics_mixed_acc_group_rate": float(mixed) / n_groups,
        "physics_reward_group_std_mean": float(sum(stds) / max(len(stds), 1)),
        "physics_reward_zero_std_rate": float(zero_std) / n_groups,
        "physics_hybrid_reward_mean": float(sum(rewards) / n),
        "physics_format_rate": format_rate,
        "physics_reward_cache_hit_rate": float(cache_hits) / n,
        "physics_reward_batch_cache_hits": float(cache_hits),
        "physics_reward_batch_unique_scored": float(unique_scored),
        "physics_outcome_only": 1.0,
    }


async def _llm_step_get_reward(req: "OpenRLHFRewardRequest") -> Dict[str, Any]:
    n = len(req.query)
    prompts = list(req.prompts) if req.prompts else [""] * n
    if len(prompts) < n:
        prompts = prompts + [""] * (n - len(prompts))
    labels = list(req.labels) if req.labels else [None] * n
    if len(labels) < n:
        labels = labels + [None] * (n - len(labels))
    if _hybrid_llm_outcome_mode() and hybrid_labels_missing(labels, n):
        raise HTTPException(
            status_code=400,
            detail="hybrid_llm_outcome requires labels; refusing process-only fallback",
        )
    questions = [str(p or "") for p in prompts[:n]]
    responses = [
        _effective_response(_response_from_query(str(query), str(prompt or "")))
        for query, prompt in zip(req.query, prompts)
    ]
    groups = group_indices_by_key(questions)
    judge = _get_llm_step_judge()
    results: List[Optional[Dict[str, Any]]] = [None] * n
    cache_hits = 0
    unique_groups = 0
    started = time.time()

    async def _score_one_group(idxs: List[int]) -> tuple[List[int], List[Dict[str, Any]], bool]:
        q = questions[idxs[0]]
        cands = [responses[i] for i in idxs]
        # Process scores are cached without gold; labels applied after the judge returns.
        cache_key = llm_step_group_cache_key(q, cands, prompt_version=judge.prompt_version)
        cached = _reward_cache.get(cache_key)
        if cached is not None and cached.get("payloads"):
            return idxs, cached["payloads"], True
        payloads = await _invoke_llm_score_group(judge, q, cands)
        _reward_cache.put(cache_key, {"payloads": payloads})
        return idxs, payloads, False

    try:
        scored = await asyncio.gather(*[_score_one_group(list(idxs)) for idxs in groups])
        for idxs, payloads, cache_hit in scored:
            if len(payloads) != len(idxs):
                raise LLMStepJudgeError("judge returned unexpected group size")
            if cache_hit:
                cache_hits += len(idxs)
            else:
                unique_groups += 1
            latency_ms = (time.time() - started) * 1000.0
            for local_i, idx in enumerate(idxs):
                results[idx] = _llm_step_payload(payloads[local_i], latency_ms=latency_ms, cache_hit=cache_hit)
    except LLMStepJudgeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if _hybrid_llm_outcome_mode():
        for idx, payload in enumerate(results):
            if payload is None:
                continue
            results[idx] = apply_hybrid_outcome(payload, responses[idx], labels[idx])

    resolved = [r or {} for r in results]
    rewards = [float(r.get("score", 0.0)) for r in resolved]
    if _hybrid_llm_outcome_mode():
        process_scores = [float((r.get("reward_components") or {}).get("llm_step") or 0.0) for r in resolved]
        accs = [1.0 if r.get("acc") else 0.0 for r in resolved]
        extra = _hybrid_group_logs(
            questions,
            rewards,
            process_scores,
            accs,
            resolved,
            cache_hits=cache_hits,
            unique_groups=unique_groups,
        )
        extra["physics_llm_step_mode"] = 0.0
        scored_by = "hybrid_llm_outcome"
    else:
        extra = _llm_step_group_logs(
            questions,
            rewards,
            resolved,
            cache_hits=cache_hits,
            unique_groups=unique_groups,
        )
        extra["physics_llm_step_mode"] = 1.0
        extra["physics_acc"] = 0.0
        extra["physics_answer_acc"] = 0.0
        scored_by = "deepseek_llm_step_score"
    _append_metrics(
        {
            "ts": time.time(),
            "scored_by": scored_by,
            "prompt_version": LLM_STEP_PROMPT_VERSION,
            "judge_model": LLM_STEP_MODEL,
            "reward_mode": REWARD_MODE,
            **{k: v for k, v in extra.items() if isinstance(v, (int, float))},
        }
    )
    return {"rewards": rewards, "scores": rewards, "extra_logs": extra}


@app.post("/get_reward")
async def openrlhf_get_reward(req: OpenRLHFRewardRequest) -> Dict[str, Any]:
    """OpenRLHF-compatible remote reward endpoint.

    Expects JSON: {query: [...], prompts: [...], labels: [...]}
    Returns: {rewards: [...], scores: [...], extra_logs: {...}}
    """
    if _uses_llm_judge():
        return await _llm_step_get_reward(req)
    n = len(req.query)
    prompts = list(req.prompts) if req.prompts else [""] * n
    labels = list(req.labels) if req.labels else [None] * n
    if len(prompts) < n:
        prompts = prompts + [""] * (n - len(prompts))
    if len(labels) < n:
        labels = labels + [None] * (n - len(labels))
    if _outcome_only_mode() and hybrid_labels_missing(labels, n):
        raise HTTPException(
            status_code=400,
            detail="outcome_only requires labels; refusing unlabeled fallback",
        )

    score_reqs = []
    for query, prompt, label in zip(req.query, prompts, labels):
        response = _response_from_query(str(query), str(prompt or ""))
        score_reqs.append(
            ScoreRequest(
                prompt=str(prompt or ""),
                response=response,
                label=label,
                question=str(prompt or ""),
            )
        )

    results: List[Optional[Dict[str, Any]]] = [None] * n
    keys = [
        reward_cache_key(_extract_question(r), _effective_response(r.response), r.label)
        for r in score_reqs
    ]
    miss_indices: List[int] = []
    cache_hits = 0
    for idx, key in enumerate(keys):
        cached = _reward_cache.get(key)
        if cached is not None:
            cached["cache_hit"] = True
            results[idx] = cached
            cache_hits += 1
        else:
            miss_indices.append(idx)

    unique_groups = group_indices_by_key([keys[i] for i in miss_indices])
    to_run: List[int] = []
    group_members: List[List[int]] = []
    for group in unique_groups:
        real_idxs = [miss_indices[i] for i in group]
        to_run.append(real_idxs[0])
        group_members.append(real_idxs)

    scored = []
    if to_run:
        scored = await asyncio.gather(*[score_one(score_reqs[i], i) for i in to_run])
    for members, payload in zip(group_members, scored):
        for idx in members:
            results[idx] = copy.deepcopy(payload)

    resolved: List[Dict[str, Any]] = [r or {} for r in results]
    rewards = [float(r.get("score", 0.0)) for r in resolved]
    accs = [1.0 if r.get("acc") else 0.0 for r in resolved]
    n_errors = [float(r.get("n_errors", 0) or 0) for r in resolved]
    verifier_hits = [1.0 if r.get("verifier_mode") == "full" else 0.0 for r in resolved]
    verifier_failed = [1.0 if (r.get("reward_components") or {}).get("verifier_failed") else 0.0 for r in resolved]
    latencies = [float(r.get("latency_ms", 0.0) or 0.0) for r in resolved]
    n_paras = [float(r.get("n_paragraphs") or 0) for r in resolved]
    n_bad = [float(r.get("n_bad_paragraphs") or 0) for r in resolved]
    cache_stats = _reward_cache.stats()
    extra: Dict[str, Any] = {
        "physics_acc": sum(accs) / max(len(accs), 1),
        "physics_n_errors_mean": sum(n_errors) / max(len(n_errors), 1),
        "physics_verifier_trigger_rate": sum(verifier_hits) / max(len(verifier_hits), 1),
        "physics_verifier_fail_rate": sum(verifier_failed) / max(len(verifier_failed), 1),
        "physics_reward_latency_ms_mean": sum(latencies) / max(len(latencies), 1),
        "physics_reward_mode": REWARD_MODE,
        "physics_answer_acc": sum(accs) / max(len(accs), 1),
        "physics_format_weight": float(_reward_weights().get("format", 0.0)),
        "physics_n_paragraphs_mean": sum(n_paras) / max(len(n_paras), 1),
        "physics_n_bad_paragraphs_mean": sum(n_bad) / max(len(n_bad), 1),
        "physics_reward_cache_hit_rate": cache_stats["hit_rate"],
        "physics_reward_batch_cache_hits": float(cache_hits),
        "physics_reward_batch_unique_scored": float(len(to_run)),
    }
    if _outcome_only_mode():
        extra.update(
            _outcome_group_logs(
                prompts,
                rewards,
                accs,
                resolved,
                cache_hits=cache_hits,
                unique_scored=len(to_run),
            )
        )
    _append_metrics(
        {
            "ts": time.time(),
            "scored_by": "outcome_only" if _outcome_only_mode() else "physics_verifier",
            "reward_mode": REWARD_MODE,
            **{k: v for k, v in extra.items() if isinstance(v, (int, float, str))},
        }
    )
    return {
        "rewards": rewards,
        "scores": rewards,
        "extra_logs": extra,
    }


def main() -> None:
    global DEFAULT_LAMBDA, DEFAULT_CAP, DEFAULT_CONCURRENCY, REWARD_MODE
    global W_ANSWER, W_FORMAT, W_VERIFIER, W_LENGTH, VERIFIER_SAMPLE_RATE
    global VERIFIER_ON_WRONG, W_PROCESS, PROCESS_ALPHA

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--lambda-penalty", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--error-cap", type=int, default=DEFAULT_CAP)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--reward-mode", default=REWARD_MODE)
    args = parser.parse_args()

    DEFAULT_LAMBDA = args.lambda_penalty
    DEFAULT_CAP = args.error_cap
    DEFAULT_CONCURRENCY = args.concurrency
    _init_runtime(DEFAULT_CONCURRENCY)
    REWARD_MODE = args.reward_mode

    if REWARD_MODE == "process_paragraph":
        VERIFIER_ON_WRONG = True
    if REWARD_MODE not in {"answer_only", "outcome_only", "llm_step_score", "hybrid_llm_outcome"}:
        _get_verifier()
    if REWARD_MODE in {"llm_step_score", "hybrid_llm_outcome"}:
        _get_llm_step_judge()
    print(
        json.dumps(
            {
                "event": "physics_reward_server_start",
                "host": args.host,
                "port": args.port,
                "lambda": DEFAULT_LAMBDA,
                "cap": DEFAULT_CAP,
                "concurrency": DEFAULT_CONCURRENCY,
                "reward_mode": REWARD_MODE,
                "weights": _reward_weights(),
                "process_only_reward": REWARD_MODE == "process_paragraph",
                "verifier_sample_rate": VERIFIER_SAMPLE_RATE,
                "openai_base_url": os.environ.get("OPENAI_BASE_URL", ""),
                "metrics_log": METRICS_LOG,
                "unified_rules": os.environ.get(
                    "PHYSICSVERIFIER_UNIFIED_RULES",
                    str(DEFAULT_UNIFIED_RULES),
                ),
                "symbolic_enabled": SYMBOLIC_ENABLED,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        limit_concurrency=max(64, DEFAULT_CONCURRENCY * 4),
        timeout_keep_alive=75,
    )


if __name__ == "__main__":
    main()
