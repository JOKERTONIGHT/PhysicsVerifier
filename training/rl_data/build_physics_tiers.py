#!/usr/bin/env python3
"""Convert PHYSICS / UGPhysics / HiPhO F=MA into Swift GRPO prompt tiers.

Writes:
  data/rl/tiers/{easy,mid,hard}_prompts.jsonl
  data/rl/tiers/eval_{easy,mid}.jsonl
  data/rl/tiers/tier_manifest.json
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = (
    "You are an expert physics solver. Show clear step-by-step reasoning "
    r"and put each final answer in \boxed{}."
)

EASY_DIFF = {
    "high school and below",
    "high school",
    "high-school",
    "gaokao",
    "f=ma",
    "introductory",
    "knowledge recall",
    "practical application",
}
MID_DIFF = {
    "high school olympiad",
    "olympiad",
    "undergraduate (non-physics major)",
    "undergraduate non-physics",
    "competition",
    "panmechanics",
    "laws application",
}
HARD_DIFF = {
    "undergraduate/postgraduate (physics major)",
    "undergraduate (physics major)",
    "graduate",
    "ipho",
    "eupho",
    "apho",
    "cpho",
    "math derivation",
}


def _load_any(path: Path, *, english_only: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if ".cache" in child.parts or child.name.startswith("."):
                continue
            if english_only and child.name.lower() in {"zh.jsonl", "zh.json"}:
                continue
            if child.suffix.lower() in {".jsonl", ".json"} and child.is_file():
                rows.extend(_load_any(child, english_only=english_only))
        return rows
    if not path.is_file():
        return rows
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return rows
    if path.suffix.lower() == ".jsonl" or "\n{" in text[:2000]:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return rows
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for key in ("data", "train", "test", "records", "examples"):
            val = obj.get(key)
            if isinstance(val, list):
                return [x for x in val if isinstance(x, dict)]
        return [obj]
    return rows


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, list):
                out.extend(_as_list(item))
            else:
                text = str(item).strip()
                if text:
                    out.append(text)
        return out
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            return _as_list(parsed)
        except Exception:
            pass
    return [text]


def _question(row: Dict[str, Any]) -> str:
    for key in ("question", "problem", "prompt", "query", "question_with_context"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def _answers(row: Dict[str, Any]) -> List[str]:
    # Prefer short gold fields. UGPhysics stores boxed finals in `answers`;
    # `solution` is a long derivation and must not become the label.
    for key in ("answer", "answers", "label", "final_answer", "gold"):
        parts = [p for p in _as_list(row.get(key)) if p and p not in {"None", "none", "null"}]
        if parts:
            return parts
    boxed = [p for p in _as_list(row.get("solution")) if "\\boxed" in str(p)]
    return boxed


def _difficulty(row: Dict[str, Any]) -> str:
    for key in ("difficulty", "level", "tier", "exam"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def _tier_of(diff: str, exam: str = "") -> str:
    blob = f"{diff} {exam}".strip().lower()
    if any(tag in blob for tag in HARD_DIFF):
        return "hard"
    if any(tag in blob for tag in MID_DIFF) or "olympiad" in blob:
        return "mid"
    if any(tag in blob for tag in EASY_DIFF) or blob.startswith("f=ma") or "high school" in blob:
        return "easy"
    return "mid"


def _to_prompt(row: Dict[str, Any], *, source: str, tier: str) -> Optional[Dict[str, Any]]:
    question = _question(row)
    answers = _answers(row)
    if not question or not answers:
        return None
    sid = str(row.get("id") or row.get("sample_id") or row.get("problem_id") or "")
    if not sid:
        sid = f"{source}:{abs(hash(question)) % 10**10}"
    solution = answers if len(answers) > 1 else answers[0]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "solution": solution,
        "question": question,
        "sample_id": sid,
        "source": source,
        "tier": tier,
        "metadata": {
            "sample_id": sid,
            "source": source,
            "tier": tier,
            "difficulty": _difficulty(row),
            "exam": str(row.get("exam") or ""),
            "answer_type": row.get("answer_type"),
        },
    }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_hipho_easy(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in _load_any(path):
        exam = str(row.get("exam") or "")
        if not re.search(r"F=?MA|PanMechanics", exam, re.I):
            continue
        tier = "easy" if re.search(r"F=?MA", exam, re.I) else "mid"
        converted = _to_prompt(row, source=f"hipho:{exam}", tier=tier)
        if converted:
            out.append(converted)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--physics-dir", type=Path, default=Path("/slow_share/jinjianhan/datasets/PHYSICS"))
    p.add_argument("--ugphysics-dir", type=Path, default=Path("/slow_share/jinjianhan/datasets/UGPhysics"))
    p.add_argument(
        "--hipho",
        type=Path,
        default=Path("/slow_share/jinjianhan/workspace/benchmarks/hipho/hipho_text_only.jsonl"),
    )
    p.add_argument("--out-dir", type=Path, default=ROOT / "data/rl/tiers")
    p.add_argument("--eval-easy-n", type=int, default=300)
    p.add_argument("--eval-mid-n", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    buckets: Dict[str, List[Dict[str, Any]]] = {"easy": [], "mid": [], "hard": []}
    sources: Dict[str, int] = {}

    def add_rows(rows: Iterable[Dict[str, Any]], source: str, default_tier: Optional[str] = None) -> None:
        n = 0
        for row in rows:
            if "tier" in row and "messages" in row:
                tier = str(row.get("tier") or default_tier or "mid")
                buckets.setdefault(tier, []).append(row)
                n += 1
                continue
            tier = default_tier or _tier_of(_difficulty(row), str(row.get("exam") or ""))
            converted = _to_prompt(row, source=source, tier=tier)
            if converted is None:
                continue
            buckets[converted["tier"]].append(converted)
            n += 1
        sources[source] = sources.get(source, 0) + n

    if args.physics_dir.exists():
        add_rows(_load_any(args.physics_dir), "PHYSICS")
    if args.ugphysics_dir.exists():
        add_rows(_load_any(args.ugphysics_dir, english_only=True), "UGPhysics")
    hipho_eval: Dict[str, List[Dict[str, Any]]] = {"easy": [], "mid": []}
    if args.hipho.exists():
        hipho_rows = convert_hipho_easy(args.hipho)
        sources["hipho"] = len(hipho_rows)
        for row in hipho_rows:
            tier = str(row.get("tier") or "easy")
            hipho_eval.setdefault(tier, []).append(row)

    rng = random.Random(args.seed)
    eval_sets: Dict[str, List[Dict[str, Any]]] = {}
    for tier, rows in buckets.items():
        rng.shuffle(rows)
        n_eval = args.eval_easy_n if tier == "easy" else args.eval_mid_n if tier == "mid" else 0
        eval_rows = rows[:n_eval] if n_eval else []
        train_rows = rows[n_eval:] if n_eval else rows
        extra = list(hipho_eval.get(tier) or [])
        buckets[tier] = train_rows
        merged_eval = eval_rows + extra
        if merged_eval:
            eval_sets[tier] = merged_eval
        _write_jsonl(args.out_dir / f"{tier}_prompts.jsonl", train_rows)

    band = ROOT / "data/rl/swift_prompts_hybrid_band.jsonl"
    if not buckets["easy"] and not buckets["mid"] and band.is_file():
        fallback_rows = []
        for row in _load_any(band):
            item = dict(row)
            item["tier"] = "mid"
            item.setdefault("source", "swift_prompts_hybrid_band")
            fallback_rows.append(item)
        buckets["mid"] = fallback_rows
        _write_jsonl(args.out_dir / "mid_prompts.jsonl", fallback_rows)
        sources["hybrid_band_fallback"] = len(fallback_rows)
    for tier, rows in eval_sets.items():
        _write_jsonl(args.out_dir / f"eval_{tier}.jsonl", rows)

    manifest = {
        "sources": sources,
        "n_train": {k: len(v) for k, v in buckets.items()},
        "n_eval": {k: len(v) for k, v in eval_sets.items()},
        "out_dir": str(args.out_dir),
        "note": "Run audit_eval_leakage.py against HiPhO and heldout before training.",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "tier_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
