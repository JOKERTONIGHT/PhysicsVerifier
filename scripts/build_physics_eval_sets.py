from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


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


def _normalize_fragment(text: str) -> str:
    t = str(text or "")
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("$", "")
    t = t.replace("\n", " ")
    t = re.sub(r"\s+", "", t)
    return t.strip().lower()


def _normalize_list(values: List[str]) -> List[str]:
    return [_normalize_fragment(v) for v in values if str(v or "").strip()]


def _parse_ground_truth_labels(item: Dict[str, Any]) -> List[str]:
    gt = item.get("ground_truth_label")
    if gt is None:
        gt = item.get("answer")
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
    tail = pred[-300:].strip()
    return [tail] if tail else []


def _strict_is_correct(item: Dict[str, Any]) -> bool:
    gt_raw = _parse_ground_truth_labels(item)
    pred_raw = _parse_prediction_answers(item)
    gt_norm = _normalize_list(gt_raw)
    pred_norm = _normalize_list(pred_raw)
    if not gt_norm:
        return False
    return gt_norm == pred_norm


def _relaxed_is_correct(item: Dict[str, Any]) -> bool:
    gt_raw = _parse_ground_truth_labels(item)
    pred_text = str(item.get("model_response") or item.get("prediction") or "")
    pred_raw = _parse_prediction_answers(item)

    gt_norm = _normalize_list(gt_raw)
    pred_norm = _normalize_list(pred_raw)
    pred_text_norm = _normalize_fragment(pred_text)

    if not gt_norm:
        return False

    # Exact list match first.
    if gt_norm == pred_norm:
        return True

    # Accept when any normalized GT fragment appears in prediction text.
    for g in gt_norm:
        if g and g in pred_text_norm:
            return True

    # Accept when any normalized prediction fragment appears in GT fragments.
    for p in pred_norm:
        if not p:
            continue
        for g in gt_norm:
            if p == g or (p in g and len(p) >= 6):
                return True
    return False


def _build_base_eval_item(item: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
    sid = str(item.get("id") or item.get("sample_index") or fallback_id)
    return {
        "id": sid,
        "question": str(item.get("question") or ""),
        "prediction": str(item.get("model_response") or item.get("prediction") or ""),
        "answer": json.dumps(item.get("ground_truth_label") or item.get("answer") or [], ensure_ascii=False),
    }


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None

    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    # Loose extraction for responses like: "Here is JSON: {...}"
    m = re.search(r"\{.*\}", raw, flags=re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _extract_errors_from_free_text(text: str, max_errors: int) -> List[str]:
    src = str(text or "").strip()
    if not src:
        return []

    lines = [re.sub(r"^[-*\d\.)\s]+", "", ln).strip() for ln in src.splitlines()]
    lines = [ln for ln in lines if ln]

    out: List[str] = []
    for ln in lines:
        if len(ln) < 16:
            continue
        # Prefer rule-like lines that clearly express constraint + violation.
        ll = ln.lower()
        if (" should " in f" {ll} " and " but " in f" {ll} ") or (
            " should " in f" {ll} " and (" not " in f" {ll} " or " violates " in f" {ll} ")
        ):
            norm = _normalize_rule_like_error(ln)
            if norm:
                out.append(norm)
        if len(out) >= max_errors:
            break
    return out


def _extract_errors_by_regex(text: str, max_errors: int) -> List[str]:
    src = str(text or "")
    if not src:
        return []

    # Extract complete JSON-like string values after "error": "..."
    matches = re.findall(r'"error"\s*:\s*"((?:\\.|[^"\\])*)"', src)
    out: List[str] = []
    for m in matches:
        try:
            decoded = json.loads(f'"{m}"')
        except Exception:
            decoded = m
        norm = _normalize_rule_like_error(decoded)
        if norm:
            out.append(norm)
        if len(out) >= max_errors:
            break
    return out


def _normalize_rule_like_error(text: str) -> str:
    s = re.sub(r"\s+", " ", str(text or "")).strip()
    if not s:
        return ""

    # Prefer rule-like phrasing that can align with checker rules.
    lower = s.lower()
    has_should = " should " in f" {lower} "
    has_but = " but " in f" {lower} "
    if has_should and has_but:
        return s

    return f"In this case, the solution should satisfy: {s}, but the answer does not satisfy this requirement."


def _extract_errors_from_payload(payload: Dict[str, Any], max_errors: int) -> List[str]:
    raw_errors = None
    if isinstance(payload, dict):
        raw_errors = payload.get("errors")
        if raw_errors is None and isinstance(payload.get("data"), list):
            raw_errors = payload.get("data")
    if not isinstance(raw_errors, list):
        return []

    out: List[str] = []
    for item in raw_errors:
        if isinstance(item, dict):
            candidate = str(item.get("error") or "").strip()
        else:
            candidate = str(item or "").strip()

        if not candidate:
            continue
        normalized = _normalize_rule_like_error(candidate)
        if normalized:
            out.append(normalized)
        if len(out) >= max_errors:
            break
    return out


def _strong_model_generation(
    *,
    model: str,
    question: str,
    prediction: str,
    answer: str,
    max_errors: int,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "ok": False,
        "errors": [],
        "raw_outputs": [],
        "attempt_logs": [],
        "last_error": "",
    }

    try:
        import os
        import openai  # type: ignore
    except Exception as e:
        result["last_error"] = f"import_error: {type(e).__name__}: {e}"
        return result

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        result["last_error"] = "missing_openai_api_key"
        return result

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)

    system_prompt = (
        "You are a strict physics evaluator for building a recall benchmark for a rule checker. "
        "Extract concise, physics-rule-grounded errors from the student answer. "
        "Each error must represent a GENERALIZABLE experience rule, not a one-off case detail. "
        "Each error should be directly mappable to a checkable rule condition. "
        "Use English only. No rubric references."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Student answer:\n{prediction}\n\n"
        f"Reference answer:\n{answer}\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        "  \"errors\": [\n"
        "    {\"error\": \"In <condition>, <expression/result> should satisfy <rule>, but the answer <violation>.\"}\n"
        "  ]\n"
        "}\n\n"
        "Style requirements for each item:\n"
        "1) Must use a 3-part rule form: CONDITION -> SHOULD RULE -> VIOLATION.\n"
        "2) CONDITION should be a reusable scenario (e.g., uniform acceleration, conservation law, boundary condition, symmetry), not a sample-specific sentence.\n"
        "3) SHOULD RULE should be a general physics relation/constraint (formula family, sign/monotonicity, conservation, dimensional consistency).\n"
        "4) VIOLATION should describe how the answer conflicts with that rule.\n"
        "5) Keep one rule violation per item.\n"
        "6) Avoid over-specific details: do NOT depend on sample id, exact numeric substitution, or one-time constants unless strictly necessary. Prefer variable-based wording.\n"
        "7) Prefer naming the rule family when possible (e.g., Newton's second law, energy conservation, continuity, boundary matching, unit consistency).\n"
        "8) Each item should be reusable as an experience rule template for similar problems.\n\n"
        "Examples (do not copy, adapt to this sample):\n"
        "- In uniform acceleration, displacement should satisfy s=v0 t+1/2 a t^2, but the answer uses a constant-velocity form s=vt.\n"
        "- For ideal gas at fixed n and T, pressure-volume should satisfy PV=const, but the answer treats P as independent of V.\n"
        "- Under valid boundary conditions, the final expression should satisfy continuity at the interface, but the answer violates that continuity condition.\n\n"
        f"Output up to {max_errors} errors."
    )

    def _repair_raw_output(raw_text: str) -> List[str]:
        raw_text = str(raw_text or "").strip()
        if not raw_text:
            return []
        repair_prompt = (
            "The previous model output may be truncated or malformed JSON. "
            "Rewrite it into valid JSON only with schema: {\"errors\":[{\"error\":\"...\"}]}. "
            "Keep only complete, meaningful, generalized rule-like physics errors. "
            "Each item must follow CONDITION -> SHOULD RULE -> VIOLATION and be reusable as an experience rule. "
            f"Limit to at most {max_errors} items.\n\n"
            f"Raw output:\n{raw_text}"
        )
        try:
            resp2 = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You repair malformed JSON outputs into strict JSON."},
                    {"role": "user", "content": repair_prompt},
                ],
                temperature=0.0,
                max_tokens=800,
                response_format={"type": "json_object"},
            )
            repaired_raw = (resp2.choices[0].message.content or "").strip()
            result["raw_outputs"].append(repaired_raw)
            repaired_payload = _extract_json_object(repaired_raw)
            if repaired_payload:
                repaired_errors = _extract_errors_from_payload(repaired_payload, max_errors=max_errors)
                if repaired_errors:
                    return repaired_errors
            return _extract_errors_by_regex(repaired_raw, max_errors=max_errors)
        except Exception as e:
            result["attempt_logs"].append({"repair_exception": f"{type(e).__name__}: {str(e)[:300]}"})
            return []

    for attempt in range(3):
        for use_json_mode in (True, False):
            kwargs: Dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 1200,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                resp = client.chat.completions.create(**kwargs)
                raw = (resp.choices[0].message.content or "").strip()
                result["raw_outputs"].append(raw)

                payload = _extract_json_object(raw)
                errors: List[str] = []
                if payload:
                    errors = _extract_errors_from_payload(payload, max_errors=max_errors)
                if not errors:
                    errors = _extract_errors_by_regex(raw, max_errors=max_errors)
                if not errors:
                    errors = _extract_errors_from_free_text(raw, max_errors=max_errors)
                if not errors and raw:
                    errors = _repair_raw_output(raw)

                result["attempt_logs"].append(
                    {
                        "attempt": attempt + 1,
                        "mode": "json_object" if use_json_mode else "plain",
                        "raw_len": len(raw),
                        "payload_parsed": bool(payload),
                        "errors_extracted": len(errors),
                    }
                )

                if errors:
                    result["ok"] = True
                    result["errors"] = errors
                    return result
            except Exception as e:
                em = f"{type(e).__name__}: {str(e)[:400]}"
                result["last_error"] = em
                result["attempt_logs"].append(
                    {
                        "attempt": attempt + 1,
                        "mode": "json_object" if use_json_mode else "plain",
                        "exception": em,
                    }
                )

        # brief backoff for transient gateway throttling/errors
        if attempt < 2:
            time.sleep(0.8)

    if not result["last_error"]:
        result["last_error"] = "no_parseable_errors_from_model_output"
    return result


def _fallback_errors(item: Dict[str, Any], max_errors: int) -> List[str]:
    pred = str(item.get("model_response") or item.get("prediction") or "")
    gt = _parse_ground_truth_labels(item)
    gt_join = "; ".join(gt) if gt else "the reference final expression"

    out = [
        f"If the final physical expression should match {gt_join}, but the answer provides a different final expression.",
        "If the derivation should stay consistent with the given physical conditions and symbols, but the answer introduces unsupported assumptions or steps.",
        "If key physical constraints in the question should be used to validate the result, but the answer does not verify the final expression against those constraints.",
    ]

    if "assume" in pred.lower() or "let" in pred.lower():
        out.insert(
            1,
            "If all assumptions should be justified by the problem statement, but the answer adds assumptions without explicit support from the given conditions.",
        )

    return out[:max_errors]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build recall/precision physics evaluation sets.")
    parser.add_argument("--input", type=str, default="data/physics_rubric_data_1000.json", help="Legacy shared input path (kept for compatibility).")
    parser.add_argument("--recall-input", type=str, default="data/evaluation_sample_1000_expansion.json")
    parser.add_argument("--precision-input", type=str, default="data/physics_rubric_data_1000.json")
    parser.add_argument("--recall-output", type=str, default="data/evaluation_recall_20.json")
    parser.add_argument("--precision-output", type=str, default="data/evaluation_precision_20.json")
    parser.add_argument("--recall-size", type=int, default=20)
    parser.add_argument("--precision-size", type=int, default=20)
    parser.add_argument("--skip-recall", action="store_true")
    parser.add_argument("--skip-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--strong-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--max-errors", type=int, default=4)
    args = parser.parse_args()

    recall_input_path = args.recall_input or args.input
    precision_input_path = args.precision_input or args.input

    recall_data = json.loads(Path(recall_input_path).read_text(encoding="utf-8"))
    if not isinstance(recall_data, list):
        raise SystemExit("Recall input file must be a JSON array.")

    precision_data = json.loads(Path(precision_input_path).read_text(encoding="utf-8"))
    if not isinstance(precision_data, list):
        raise SystemExit("Precision input file must be a JSON array.")

    strict_correct_pool: List[Dict[str, Any]] = []
    relaxed_correct_pool: List[Dict[str, Any]] = []
    wrong_pool: List[Dict[str, Any]] = []
    for row in precision_data:
        if not isinstance(row, dict):
            continue
        if _strict_is_correct(row):
            strict_correct_pool.append(row)
            relaxed_correct_pool.append(row)
        elif _relaxed_is_correct(row):
            relaxed_correct_pool.append(row)
        else:
            wrong_pool.append(row)

    rng = random.Random(args.seed)
    recall_take = min(args.recall_size, len(recall_data))
    precision_take = min(args.precision_size, len(relaxed_correct_pool))

    recall_rows = rng.sample(recall_data, recall_take) if recall_take > 0 else []
    precision_rows = rng.sample(relaxed_correct_pool, precision_take) if precision_take > 0 else []

    recall_out: List[Dict[str, Any]] = []
    llm_used = 0
    llm_failed = 0
    failure_reason_counter: Dict[str, int] = {}
    if not args.skip_recall:
        for i, row in enumerate(recall_rows):
            base = _build_base_eval_item(row, fallback_id=f"recall_{i}")
            gen = _strong_model_generation(
                model=args.strong_model,
                question=base["question"],
                prediction=base["prediction"],
                answer=base["answer"],
                max_errors=args.max_errors,
            )
            errors = gen.get("errors") if isinstance(gen.get("errors"), list) else []
            if gen.get("ok") and errors:
                source = "strong_model"
                llm_used += 1
            else:
                source = "strong_model_failed"
                llm_failed += 1
                reason = str(gen.get("last_error") or "unknown_failure")
                failure_reason_counter[reason] = failure_reason_counter.get(reason, 0) + 1

            recall_out.append(
                {
                    **base,
                    "physics_error_examples": [{"error": e} for e in errors],
                    "error_source": source,
                    "strong_model_raw_outputs": gen.get("raw_outputs") or [],
                    "strong_model_attempt_logs": gen.get("attempt_logs") or [],
                    "strong_model_last_error": str(gen.get("last_error") or ""),
                    "expected_has_physics_error": True,
                }
            )

    precision_out: List[Dict[str, Any]] = []
    if not args.skip_precision:
        for i, row in enumerate(precision_rows):
            base = _build_base_eval_item(row, fallback_id=f"precision_{i}")
            precision_out.append(
                {
                    **base,
                    "expected_has_physics_error": False,
                }
            )

    recall_path = Path(args.recall_output)
    precision_path = Path(args.precision_output)
    recall_path.parent.mkdir(parents=True, exist_ok=True)
    precision_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_recall:
        recall_path.write_text(json.dumps(recall_out, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.skip_precision:
        precision_path.write_text(json.dumps(precision_out, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input": args.input,
        "recall_input": recall_input_path,
        "precision_input": precision_input_path,
        "recall_total_samples": len(recall_data),
        "precision_total_samples": len(precision_data),
        "strict_correct_pool": len(strict_correct_pool),
        "relaxed_correct_pool": len(relaxed_correct_pool),
        "strict_wrong_pool": len(wrong_pool),
        "recall_size": len(recall_out) if not args.skip_recall else 0,
        "precision_size": len(precision_out) if not args.skip_precision else 0,
        "skip_recall": bool(args.skip_recall),
        "skip_precision": bool(args.skip_precision),
        "strong_model": args.strong_model,
        "recall_error_source": {
            "strong_model": llm_used,
            "strong_model_failed": llm_failed,
        },
        "recall_failure_reasons": failure_reason_counter,
        "recall_output": str(recall_path) if not args.skip_recall else None,
        "precision_output": str(precision_path) if not args.skip_precision else None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
