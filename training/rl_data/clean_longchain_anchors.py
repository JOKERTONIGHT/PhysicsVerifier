#!/usr/bin/env python3
"""Clean base_rollout_anchor rows in the long-chain SFT pool.

Strips markdown headings and emoji, keeps only the last \\boxed{}, and
drops repetitive tails. Re-checks gold with answers_equivalent. Failed
anchors are discarded; API/other generators are copied through.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.rl_data.answer_equiv import answers_equivalent
from training.rl_data.screen_training_data import (
    EMOJI_RE,
    MD_HEADING_RE,
    _looks_repetitive_tail,
    assistant_text,
    sample_id,
)

BLANK_RE = re.compile(r"\n{3,}")
HEADING_LINE_RE = re.compile(r"(?m)^#{1,6}\s.*$")
BOXED_NEEDLE = "\\boxed"


def boxed_spans(text: str) -> List[Tuple[int, int, str]]:
    """Return (start, end, inner) for each \\boxed{...} with brace matching."""
    spans: List[Tuple[int, int, str]] = []
    start = 0
    src = text or ""
    while True:
        idx = src.find(BOXED_NEEDLE, start)
        if idx < 0:
            break
        j = idx + len(BOXED_NEEDLE)
        while j < len(src) and src[j].isspace():
            j += 1
        if j >= len(src) or src[j] != "{":
            start = j
            continue
        depth = 0
        k = j
        while k < len(src):
            if src[k] == "{":
                depth += 1
            elif src[k] == "}":
                depth -= 1
                if depth == 0:
                    spans.append((idx, k + 1, src[j + 1 : k].strip()))
                    break
            k += 1
        start = k + 1 if k < len(src) else j + 1
    return spans


def keep_last_boxed(text: str) -> str:
    spans = boxed_spans(text)
    if len(spans) <= 1:
        return text
    parts: List[str] = []
    last = 0
    for i, (s, e, inner) in enumerate(spans):
        parts.append(text[last:s])
        parts.append(text[s:e] if i == len(spans) - 1 else inner)
        last = e
    parts.append(text[last:])
    return "".join(parts)


def trim_after_last_boxed(text: str) -> str:
    spans = boxed_spans(text)
    if not spans:
        return text
    return text[: spans[-1][1]].rstrip() + "\n"


def looks_repetitive_body(text: str, window: int = 80, min_repeats: int = 5) -> bool:
    if _looks_repetitive_tail(text):
        return True
    src = text or ""
    if len(src) < window * min_repeats:
        return False
    for frac in (0.25, 0.5, 0.75):
        i = int(len(src) * frac)
        needle = src[max(0, i) : i + window]
        if needle.strip() and src.count(needle) >= min_repeats:
            return True
    return False


def drop_duplicate_paragraphs(text: str) -> str:
    paras = re.split(r"\n{2,}", text or "")
    out: List[str] = []
    for p in paras:
        if out and p.strip() == out[-1].strip():
            continue
        out.append(p)
    return "\n\n".join(out)


def drop_repetitive_tail(text: str, window: int = 200, min_repeats: int = 3) -> str:
    src = text or ""
    while _looks_repetitive_tail(src, window=window, min_repeats=min_repeats):
        needle = src[-window:]
        idx = src.rfind(needle)
        if idx <= 0:
            break
        src = src[:idx].rstrip()
        if len(src) < window * min_repeats:
            break
    return src


def clean_assistant(text: str) -> str:
    src = HEADING_LINE_RE.sub("", text or "")
    src = EMOJI_RE.sub("", src)
    src = keep_last_boxed(src)
    src = trim_after_last_boxed(src)
    src = drop_repetitive_tail(src)
    src = drop_duplicate_paragraphs(src)
    src = BLANK_RE.sub("\n\n", src).strip()
    if src and not src.endswith("\n"):
        src += "\n"
    return src


def gold_text(row: Dict[str, Any]) -> str:
    for key in ("solution", "label", "answer"):
        val = row.get(key)
        if isinstance(val, list):
            parts = [str(x) for x in val if x is not None and str(x).strip()]
            if parts:
                return parts[0]
        if val is not None and str(val).strip():
            return str(val)
    meta = row.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("label"):
        return str(meta["label"])
    return ""


def replace_assistant(row: Dict[str, Any], text: str) -> Dict[str, Any]:
    out = dict(row)
    msgs = list(row.get("messages") or [])
    if msgs:
        replaced = False
        new_msgs: List[Dict[str, Any]] = []
        for msg in reversed(msgs):
            item = dict(msg)
            if not replaced and item.get("role") == "assistant":
                item["content"] = text
                replaced = True
            new_msgs.append(item)
        out["messages"] = list(reversed(new_msgs))
    if "output" in row:
        out["output"] = text
    return out


def drop_reason(
    cleaned: str,
    gold: str,
    *,
    min_chars: int,
) -> Optional[str]:
    if not cleaned.strip():
        return "empty"
    if MD_HEADING_RE.search(cleaned):
        return "md_heading"
    if EMOJI_RE.search(cleaned):
        return "emoji"
    n_box = cleaned.count(BOXED_NEEDLE)
    if n_box == 0:
        return "no_boxed"
    if n_box >= 2:
        return "multi_box"
    if looks_repetitive_body(cleaned):
        return "repetition"
    if min_chars and len(cleaned) < min_chars:
        return "too_short"
    if gold:
        ok, _why = answers_equivalent(cleaned, gold)
        if not ok:
            return "gold_mismatch"
    return None


def clean_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    min_chars: int = 2500,
    generators: Sequence[str] = ("base_rollout_anchor",),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    want = set(generators)
    kept: List[Dict[str, Any]] = []
    dropped: Dict[str, int] = {}
    n_anchor = 0
    n_cleaned = 0
    n_passthrough = 0
    for row in rows:
        gen = str(row.get("generator") or row.get("source") or "")
        if gen not in want:
            kept.append(row)
            n_passthrough += 1
            continue
        n_anchor += 1
        cleaned = clean_assistant(assistant_text(row))
        reason = drop_reason(cleaned, gold_text(row), min_chars=min_chars)
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        out = replace_assistant(row, cleaned)
        out["generator"] = "base_rollout_anchor_clean"
        out["cleaned_from"] = gen
        kept.append(out)
        n_cleaned += 1
    report = {
        "n_in": len(rows),
        "n_out": len(kept),
        "n_anchor": n_anchor,
        "n_cleaned": n_cleaned,
        "n_dropped": n_anchor - n_cleaned,
        "n_passthrough": n_passthrough,
        "dropped_reasons": dropped,
        "min_chars": min_chars,
    }
    return kept, report


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=ROOT / "data/rl/sft_solutions_longchain.jsonl")
    p.add_argument("--output", type=Path, default=ROOT / "data/rl/sft_solutions_longchain_clean.jsonl")
    p.add_argument("--report", type=Path, default=ROOT / "data/rl/sft_longchain_clean_report.json")
    p.add_argument("--min-chars", type=int, default=2500)
    args = p.parse_args()
    rows = _load_jsonl(args.input)
    kept, report = clean_rows(rows, min_chars=args.min_chars)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
