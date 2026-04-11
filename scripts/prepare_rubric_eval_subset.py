from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _extract_boxed(text: str) -> List[str]:
    src = str(text or "")
    out: List[str] = []
    i = 0
    token = "\\boxed{"
    n = len(src)

    while i < n:
        j = src.find(token, i)
        if j < 0:
            break
        k = j + len(token)
        depth = 1
        start = k
        while k < n and depth > 0:
            ch = src[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            k += 1
        if depth == 0:
            out.append(src[start:k - 1])
            i = k
        else:
            break
    return out


def _normalize_answer_fragment(text: str) -> str:
    t = str(text or "")
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("$", "")
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", "", t)
    return t.strip().lower()


def _normalize_answer_list(values: List[str]) -> List[str]:
    return [_normalize_answer_fragment(v) for v in values if str(v or "").strip()]


def _parse_ground_truth_labels(item: Dict[str, Any]) -> List[str]:
    gt = item.get("ground_truth_label")
    if isinstance(gt, list):
        out: List[str] = []
        for x in gt:
            s = str(x or "")
            boxed = _extract_boxed(s)
            if boxed:
                out.extend(boxed)
            elif s.strip():
                out.append(s)
        return out
    if gt is not None:
        s = str(gt)
        boxed = _extract_boxed(s)
        if boxed:
            return boxed
        if s.strip():
            return [s]
    return []


def _parse_prediction_answers(item: Dict[str, Any]) -> List[str]:
    pred = str(item.get("model_response") or item.get("prediction") or "")
    boxed = _extract_boxed(pred)
    if boxed:
        return boxed
    # Fall back to a conservative single fragment when no box exists.
    tail = pred[-300:].strip()
    return [tail] if tail else []


def _build_eval_item(item: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    sid = str(item.get("id") or item.get("sample_index") or fallback_id)
    return {
        "id": sid,
        "question": str(item.get("question") or ""),
        "prediction": str(item.get("model_response") or item.get("prediction") or ""),
        "answer": json.dumps(item.get("ground_truth_label") or item.get("answer") or [], ensure_ascii=False),
    }


def _build_meta_item(item: Dict[str, Any], eval_id: str) -> Dict[str, Any]:
    gt_raw = _parse_ground_truth_labels(item)
    pred_raw = _parse_prediction_answers(item)

    gt_norm = _normalize_answer_list(gt_raw)
    pred_norm = _normalize_answer_list(pred_raw)

    is_evaluable = bool(gt_norm)
    strict_exact_match: Optional[bool] = None
    gt_is_incorrect: Optional[bool] = None
    if is_evaluable:
        strict_exact_match = gt_norm == pred_norm
        gt_is_incorrect = not strict_exact_match

    return {
        "id": eval_id,
        "ground_truth_label": item.get("ground_truth_label") or [],
        "rubric_list": item.get("rubric_list") or [],
        "answer_type": item.get("answer_type") or [],
        "ground_truth_normalized": gt_norm,
        "prediction_normalized": pred_norm,
        "is_evaluable": is_evaluable,
        "strict_exact_match": strict_exact_match,
        "gt_is_incorrect": gt_is_incorrect,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small rubric evaluation subset and a strict-metric meta file.",
    )
    parser.add_argument("--input", type=str, default="data/physics_rubric_data_1000.json")
    parser.add_argument("--output-eval", type=str, default="data/evaluation_rubric_100.json")
    parser.add_argument("--output-meta", type=str, default="data/rubric_eval_100_meta.json")
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260331)
    args = parser.parse_args()

    if args.size <= 0:
        raise SystemExit("--size must be > 0")

    in_path = Path(args.input)
    eval_path = Path(args.output_eval)
    meta_path = Path(args.output_meta)

    data = json.loads(in_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit("Input rubric file must be a JSON array.")

    if args.size > len(data):
        raise SystemExit(f"Requested size={args.size} exceeds dataset size={len(data)}")

    rng = random.Random(args.seed)
    chosen_indices = sorted(rng.sample(range(len(data)), args.size))

    eval_items: List[Dict[str, Any]] = []
    meta_items: List[Dict[str, Any]] = []

    for idx in chosen_indices:
        raw = data[idx]
        if not isinstance(raw, dict):
            continue
        fallback_id = f"rubric_{idx}"
        eval_item = _build_eval_item(raw, fallback_id)
        sid = str(eval_item.get("id") or fallback_id)
        eval_items.append(eval_item)
        meta_items.append(_build_meta_item(raw, sid))

    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_path.write_text(json.dumps(eval_items, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_payload = {
        "summary": {
            "input": str(in_path),
            "size": len(meta_items),
            "seed": args.seed,
            "evaluable_count": len([m for m in meta_items if m.get("is_evaluable") is True]),
        },
        "samples": meta_items,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(meta_payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
