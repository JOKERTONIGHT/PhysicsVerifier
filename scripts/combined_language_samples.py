"""Normalize rollout samples from `combined_language_only`-style JSON into verifier/eval rows."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from typing import Any, Dict, List, Optional, Tuple


def parse_reward_field(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def parse_metadata_field(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def extract_question(sample: Dict[str, Any]) -> str:
    md = parse_metadata_field(sample.get("metadata"))
    q = md.get("question") or md.get("stem") or md.get("title")
    if q:
        return str(q).strip()
    prompt = str(sample.get("prompt") or "")
    m = re.search(r"User:\s*(.*?)(?=\nAssistant:|<Assistant|\Z)", prompt, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    return prompt.strip()


def stable_question_key(question: str) -> str:
    raw = " ".join(str(question or "").split())[:4000].strip().casefold()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def sample_to_eval_row(
    rollout_id: Any,
    sample: Dict[str, Any],
    *,
    seq_index: int,
) -> Dict[str, Any]:
    q = extract_question(sample)
    sid = str(sample.get("index") if sample.get("index") is not None else seq_index)
    uid = f"cl_{rollout_id}_{sid}"
    reward = parse_reward_field(sample.get("reward"))
    label = sample.get("label")
    answer = json.dumps(label if isinstance(label, list) else ([label] if label is not None else []), ensure_ascii=False)
    row: Dict[str, Any] = {
        "id": uid,
        "question": q,
        "prediction": str(sample.get("response") or ""),
        "answer": answer,
        "source_reward_score": reward.get("score"),
        "source_reward_acc": reward.get("acc") if isinstance(reward.get("acc"), bool) else None,
        "meta": {
            "rollout_id": rollout_id,
            "sample_index": sample.get("index"),
            "response_length": sample.get("response_length"),
            "reward_score": reward.get("score"),
            "reward_acc": reward.get("acc"),
            "question_key": stable_question_key(q),
            "status": sample.get("status"),
        },
    }
    return row


def reservoir_sample_indices(rng, stream_len: int, k: int) -> List[int]:
    """Return sorted indices to keep for reservoir size k (algorithm selects which positions survive)."""
    import random

    if k <= 0 or stream_len <= 0:
        return []
    if stream_len <= k:
        return list(range(stream_len))

    pool = list(range(k))
    for i in range(k, stream_len):
        j = rng.randint(0, i)
        if j < k:
            pool[j] = i
    return sorted(pool)


def reservoir_mask_for_stream(rng, items_iter, k: int) -> List[Tuple[Any, Dict[str, Any]]]:
    """Collect exactly k items uniformly from an iterable of unknown length (one-pass)."""
    import random

    pool: List[Tuple[Any, Dict[str, Any]]] = []
    n = 0
    for key, row in items_iter:
        n += 1
        if len(pool) < k:
            pool.append((key, row))
            continue
        j = rng.randint(0, n - 1)
        if j < k:
            pool[j] = (key, row)
    return pool
