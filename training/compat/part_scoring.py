"""Single source of truth for part-level physics answer scoring.

Both the HiPhO/heldout scorer and the RL reward server must import from here.
Do not reimplement boxed extraction or part matching in either caller.
"""
from __future__ import annotations

import math
import random
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from training.compat.math_grading import extract_answer, grade_answer_verl
from training.rl_data.answer_equiv import answers_equivalent

END_PUNCT_RE = re.compile(r"[.。!?）)\}\]$]$")


def extract_all_boxed(text: str) -> List[str]:
    """Return inner contents of every ``\\boxed{}`` / ``\\fbox{}`` in order."""
    src = str(text or "")
    out: List[str] = []
    start = 0
    while start < len(src):
        idx_box = src.find("\\boxed", start)
        idx_fbox = src.find("\\fbox", start)
        candidates = [i for i in (idx_box, idx_fbox) if i >= 0]
        if not candidates:
            break
        idx = min(candidates)
        cmd_len = 6 if src.startswith("\\boxed", idx) else 5
        i = idx + cmd_len
        while i < len(src) and src[i].isspace():
            i += 1
        if i >= len(src):
            break
        if src[i] != "{":
            j = i
            while j < len(src) and (not src[j].isspace()) and src[j] not in "\\$":
                j += 1
            if j > i:
                out.append(src[i:j])
            start = max(j, i + 1)
            continue
        depth = 0
        j = i
        while j < len(src):
            if src[j] == "{":
                depth += 1
            elif src[j] == "}":
                depth -= 1
                if depth == 0:
                    out.append(src[i + 1 : j])
                    start = j + 1
                    break
            j += 1
        else:
            break
    return out


def gold_parts(labels: Sequence[str]) -> List[str]:
    parts: List[str] = []
    for lab in labels:
        boxed = extract_all_boxed(lab)
        if boxed:
            parts.extend(p.strip() for p in boxed if str(p).strip())
            continue
        cleaned = str(lab).strip().strip("$").strip()
        if cleaned:
            parts.append(cleaned)
    return parts


def looks_repetitive(text: str, window: int = 200, min_repeats: int = 3) -> bool:
    src = str(text or "")
    if len(src) < window * min_repeats:
        return False
    needle = src[-window:]
    if not needle.strip():
        return False
    return src.count(needle) >= min_repeats


def boxed_unclosed(text: str) -> bool:
    src = str(text or "")
    idx = max(src.rfind("\\boxed"), src.rfind("\\fbox"))
    if idx < 0:
        return False
    i = idx
    while i < len(src) and src[i] not in "{":
        i += 1
    if i >= len(src) or src[i] != "{":
        return False
    depth = 0
    while i < len(src):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return False
        i += 1
    return True


def looks_truncated(text: str) -> bool:
    src = str(text or "").rstrip()
    if not src:
        return True
    if boxed_unclosed(src):
        return True
    if "\\boxed" not in src and "\\fbox" not in src:
        return not bool(END_PUNCT_RE.search(src))
    return False


def as_boxed(text: str) -> str:
    src = str(text or "").strip()
    if "\\boxed" in src or "\\fbox" in src:
        return src if src.startswith("\\") else f"\\boxed{{{src}}}"
    return f"\\boxed{{{src}}}"


def answers_match(cand: str, gold: str) -> bool:
    if not cand or not gold:
        return False
    cand_box = as_boxed(cand)
    gold_box = as_boxed(gold)
    if grade_answer_verl(cand_box, gold_box):
        return True
    ok, _ = answers_equivalent(cand_box, gold_box)
    return bool(ok)


def score_prediction(pred: str, labels: Sequence[str]) -> Dict[str, Any]:
    pred = str(pred or "")
    boxes = extract_all_boxed(pred)
    parts = gold_parts(labels)
    hits = 0
    for part in parts:
        if any(answers_match(box, part) for box in boxes):
            hits += 1
            continue
        if boxes and answers_match(pred, part):
            hits += 1
    item_correct = bool(parts) and hits == len(parts)
    part_frac = (hits / len(parts)) if parts else 0.0
    extracted_pred = boxes[-1] if boxes else (extract_answer(pred) or "")
    extracted_gt = parts[0] if parts else ""
    return {
        "n_parts": len(parts),
        "n_hit": hits,
        "part_frac": part_frac,
        "item_correct": item_correct,
        "n_boxed": len(boxes),
        "no_boxed": len(boxes) == 0,
        "truncated": looks_truncated(pred),
        "repetitive": looks_repetitive(pred),
        "extracted_pred": extracted_pred,
        "extracted_gt": extracted_gt,
        "gold_parts": parts,
        "pred_boxes": boxes,
    }


def labels_from_value(answer: Any) -> List[str]:
    if answer is None:
        return []
    if isinstance(answer, list):
        return [str(x) for x in answer if x is not None and str(x).strip()]
    return [str(answer)]


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Percentile bootstrap CI for a mean. Also returns a binomial-style SE."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return {"mean": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0, "n": 0.0, "n_boot": float(n_boot)}
    mean = sum(xs) / n
    se = math.sqrt(sum((x - mean) ** 2 for x in xs) / max(n, 1) / max(n, 1))
    if n == 1:
        return {"mean": mean, "se": 0.0, "lo": mean, "hi": mean, "n": 1.0, "n_boot": float(n_boot)}
    rng = random.Random(seed)
    means: List[float] = []
    for _ in range(max(1, n_boot)):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_i = max(0, int(math.floor((alpha / 2.0) * len(means))))
    hi_i = min(len(means) - 1, int(math.ceil((1.0 - alpha / 2.0) * len(means))) - 1)
    return {
        "mean": mean,
        "se": se,
        "lo": means[lo_i],
        "hi": means[hi_i],
        "n": float(n),
        "n_boot": float(n_boot),
    }


def paired_delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Any]:
    """Bootstrap CI of mean(a - b) for paired per-item scores (same items, same k)."""
    n = min(len(a), len(b))
    diffs = [float(a[i]) - float(b[i]) for i in range(n)]
    ci = bootstrap_mean_ci(diffs, n_boot=n_boot, alpha=alpha, seed=seed)
    mean = ci["mean"]
    significant = (ci["lo"] > 0.0) or (ci["hi"] < 0.0)
    return {
        **ci,
        "n_paired": n,
        "significant_2se": bool(significant),
        "direction": "a>b" if mean > 0 else ("a<b" if mean < 0 else "tie"),
    }


def group_item_scores(
    recs_by_item: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[float], List[float], List[float], List[float]]:
    part_avgs: List[float] = []
    item_avgs: List[float] = []
    item_pass: List[float] = []
    part_pass: List[float] = []
    for recs in recs_by_item.values():
        part_avgs.append(sum(float(r["part_frac"]) for r in recs) / max(len(recs), 1))
        item_avgs.append(sum(float(r["item_correct"]) for r in recs) / max(len(recs), 1))
        item_pass.append(1.0 if any(r["item_correct"] for r in recs) else 0.0)
        part_pass.append(max((float(r["part_frac"]) for r in recs), default=0.0))
    return part_avgs, item_avgs, item_pass, part_pass
