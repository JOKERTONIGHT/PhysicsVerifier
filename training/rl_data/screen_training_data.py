#!/usr/bin/env python3
"""Screen physics prompts for text-only SFT/RL: drop visual-input and unusable stems.

The evaluation dumps have no image files; many stems still say "see Figure N" and
cannot be solved from the remaining text. Use this module before SFT generation
and RL training so both stages see the same usable pool.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NUMBERED_RE = re.compile(
    r"(?:(?:^|\n)\s*(?:\(?[1-9]\)|[1-9]\.|[A-Da-d][\.\)])\s+\S)"
    r"|(?:\b(?:Part|Task|Question)\s+[A-D1-9]\b)"
    r"|(?:\([1-9]\)\s)",
    re.IGNORECASE,
)
MULTI_ASK_RE = re.compile(
    r"\b(?:find|determine|calculate|what are|evaluate)\b",
    re.IGNORECASE,
)
# Phrases that contain "figure" but are not a missing drawing.
ALLOW_RE = re.compile(
    r"significant figures?|figure of merit|fig(?:ure)?\.?\s*of merit",
    re.IGNORECASE,
)
VISUAL_RE = re.compile(
    r"""
    (?:
        as\s+(?:shown|indicated|illustrated|depicted)\s+in
            \s+(?:the\s+)?(?:(?:right|left|following)\s+)?(?:figure|fig\.?|diagram|graph|photo|sketch|drawing|circuit)
      | as\s+(?:shown|indicated)\s+in.{0,40}(?:figure|fig\.?|diagram|graph)
      | (?:shown|indicated|illustrated)\s+in\s+(?:the\s+)?(?:figure|fig\.?|diagram|graph)
      | see\s+(?:the\s+)?(?:following\s+)?(?:figure|fig\.?|diagram|graph)
      | according\s+to\s+(?:the\s+)?(?:figure|diagram|graph)
      | from\s+the\s+(?:figure|diagram|graph|plot)
      | the\s+(?:figure|diagram|graph)\s+(?:below|above|on\s+the\s+(?:left|right)|shows)
      | \bfig(?:ure)?\.?\s*\d
      | \bfigures?\s+\d
      | \n\s*figures?\s*\d*\s*(?:[:.\n]|is\b)
      | circuit\s+shown
      | shown\s+(?:below|above)
      | extra\s+sheet
      | larger\s+copy
      | !\[[^\]]*\]\(
      | \.(?:png|jpe?g|gif|svg|webp)\b
      | 如图|见图|如图所示|示意图
      | I\s*[-–]\s*V\s+(?:characteristic|curve)
      | read\s+(?:off|from)\s+the\s+(?:graph|figure|diagram)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
CONCAT_RE = re.compile(
    r"Question:.{80,}Question:",
    re.DOTALL,
)
GOLD_FIT_RE = re.compile(
    r"(?:the reference (?:says|has|result tells|final answer)|"
    r"accept reference result|"
    r"instruction to (?:match|derive) the reference|"
    r"diagram \(not shown|"
    r"figure \(not shown|"
    r"without the explicit (?:diagram|circuit|figure)|"
    r"we are not given.{0,120}(?:but the known|implies we|the reference)|"
    r"invent.{0,40}typical|"
    r"matching reference form|"
    r"check with reference)",
    re.IGNORECASE | re.DOTALL,
)
MD_HEADING_RE = re.compile(r"(?m)^#{1,6}\s")
EMOJI_RE = re.compile(r"[\u2705\U0001F4CC\U0001F300-\U0001F9FF]")
QQUAD_RE = re.compile(r"\\qquad")
END_PUNCT_RE = re.compile(r"[.。!?）)\}\]$]$")

# Hand-audited stems that look text-only but still cannot be solved, or whose
# labeled solution reverse-engineers gold. Kept here so SFT rebuilds stay dry.
KNOWN_UNUSABLE_IDS = {
    "154795",
    "162_917",
    "181_506",
    "228_537",
    "244_792",
    "244_911",
    "247_840",
    "268_692",
    "99596",
    "220_385",
    "264_81",
    "219_87",
    "83_148",
    "154_313",
    "246_1",
    "84_794",
    "222_364",
    "156_1019",
    "257_426",
    "125_959",
    "256_947",
    "251_322",
    "91_1016",
    "232_110",
    "133_545",
    "230_931",
    "82_624",
    "110_952",
    "113_88",
    "138_259",
    "196_995",
    "189_545",
    "146_700",
    "100_135",
    "268_152",
    "108_114",
    "132476",  # two unrelated numbered problems glued (balloon + thermal radiation)
}


def question_text(row: Dict[str, Any]) -> str:
    q = row.get("question")
    if isinstance(q, str) and q.strip():
        return q
    meta = row.get("metadata")
    if isinstance(meta, dict):
        mq = meta.get("question")
        if isinstance(mq, str) and mq.strip():
            return mq
    for key in ("messages", "input"):
        msgs = row.get(key)
        if not isinstance(msgs, list):
            continue
        parts = []
        for msg in msgs:
            if isinstance(msg, dict) and msg.get("role") == "user":
                parts.append(str(msg.get("content") or ""))
        if parts:
            return "\n".join(parts)
    return str(row.get("input") or "")


def assistant_text(row: Dict[str, Any]) -> str:
    for key in ("messages", "input"):
        msgs = row.get(key)
        if not isinstance(msgs, list):
            continue
        for msg in reversed(msgs):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return str(msg.get("content") or "")
    return str(row.get("output") or "")


def sample_id(row: Dict[str, Any]) -> str:
    sid = row.get("sample_id")
    if sid:
        return str(sid)
    meta = row.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("sample_id"):
        return str(meta["sample_id"])
    return str(row.get("id") or "")


def _labels(row: Dict[str, Any]) -> List[str]:
    for key in ("solution", "label", "answer"):
        val = row.get(key)
        if val is None and key == "label":
            continue
        if isinstance(val, list):
            return [str(x) for x in val if x is not None and str(x).strip()]
        if val is not None and str(val).strip():
            return [str(val)]
    meta = row.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("label"):
        return [str(meta["label"])]
    return []


def visual_drop_reason(text: str) -> Optional[str]:
    """Return a reason if the stem depends on a missing figure/graph/photo."""
    stripped = ALLOW_RE.sub(" ", text or "")
    if VISUAL_RE.search(stripped):
        return "visual_input"
    return None


def looks_multi_question(question: str, n_labels: int) -> bool:
    markers = NUMBERED_RE.findall(question or "")
    unique_markers = {m.strip().lower() for m in markers}
    if len(unique_markers) >= 2 and n_labels <= 1:
        return True
    asks = MULTI_ASK_RE.findall(question or "")
    if len(asks) >= 3 and n_labels <= 1 and len(question) > 1200:
        return True
    return False


def looks_concatenated(question: str) -> bool:
    return bool(CONCAT_RE.search(question or ""))


def gold_fit_drop_reason(assistant: str) -> Optional[str]:
    if GOLD_FIT_RE.search(assistant or ""):
        return "gold_fit"
    return None


def looks_multi_blank(question: str) -> bool:
    return len(QQUAD_RE.findall(question or "")) >= 2


def _looks_repetitive_tail(text: str, window: int = 200, min_repeats: int = 3) -> bool:
    src = text or ""
    if len(src) < window * min_repeats:
        return False
    needle = src[-window:]
    if not needle.strip():
        return False
    return src.count(needle) >= min_repeats


def sft_style_drop_reason(
    text: str,
    *,
    min_chars: int = 400,
    max_chars: int = 4000,
    mode: str = "sft",
) -> Optional[str]:
    """Hard style gate for stored SFT assistant text.

    ``mode="rft"`` allows multiple ``\\boxed{}`` (multi-part questions), a
    higher length cap, and Qwen's native markdown/emoji, but still drops
    truncated / looping completions.
    """
    src = text or ""
    rft = mode == "rft"
    if rft:
        max_chars = max(max_chars, 8000)
    n_box = src.count("\\boxed")
    if n_box == 0:
        return "no_boxed"
    if not rft and n_box >= 2:
        return "multi_box"
    idx = src.rfind("\\boxed")
    if (not rft) and idx >= 0 and len(src) - idx > 400:
        return "ramble_after_box"
    if not END_PUNCT_RE.search(src.rstrip()):
        return "truncated"
    if src.count("$$") % 2:
        return "broken_latex"
    if len(src) > max_chars:
        return "too_long"
    if min_chars and len(src) < min_chars:
        return "too_short"
    if not rft and MD_HEADING_RE.search(src):
        return "md_heading"
    if not rft and EMOJI_RE.search(src):
        return "emoji"
    if rft and _looks_repetitive_tail(src):
        return "repetition"
    return None


def prompt_drop_reason(
    row: Dict[str, Any],
    *,
    n_labels: Optional[int] = None,
) -> Optional[str]:
    """Reasons that apply to any training prompt (SFT or RL)."""
    sid = sample_id(row)
    if sid and sid in KNOWN_UNUSABLE_IDS:
        return "known_unusable"
    question = question_text(row)
    if not question.strip():
        return "empty_prompt"
    visual = visual_drop_reason(question)
    if visual:
        return visual
    if looks_concatenated(question):
        return "concatenated_stem"
    if n_labels is not None and looks_multi_question(question, n_labels):
        return "multi_question_single_label"
    return None


def sft_row_drop_reason(row: Dict[str, Any]) -> Optional[str]:
    reason = prompt_drop_reason(row)
    if reason:
        return reason
    if looks_multi_blank(question_text(row)):
        return "multi_blank_stem"
    gold_fit = gold_fit_drop_reason(assistant_text(row))
    if gold_fit:
        return gold_fit
    style = sft_style_drop_reason(assistant_text(row))
    if style and style not in {"too_short"}:
        return style
    return None


def filter_rows(
    rows: Iterable[Dict[str, Any]],
    *,
    mode: str = "prompts",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    drop_fn = sft_row_drop_reason if mode == "sft" else prompt_drop_reason
    n_in = 0
    for row in rows:
        n_in += 1
        reason = drop_fn(row)
        if reason:
            rec = dict(row)
            rec["screen_reason"] = reason
            dropped.append(rec)
            reasons[reason] += 1
        else:
            kept.append(row)
    audit = {
        "n_in": n_in,
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "drop_reasons": dict(reasons),
        "mode": mode,
    }
    return kept, dropped, audit


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rejected", type=Path, default=None)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--mode", choices=["prompts", "sft"], default="prompts")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    rows = _load_jsonl(args.input)
    kept, dropped, audit = filter_rows(rows, mode=args.mode)
    audit["input"] = str(args.input)
    audit["output"] = str(args.output)
    if args.dry_run:
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return 0
    _write_jsonl(args.output, kept)
    if args.rejected:
        _write_jsonl(args.rejected, dropped)
    report_path = args.report or args.output.with_suffix(".screen.json")
    report_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
