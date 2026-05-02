from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


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


def _collapse_text_for_match(text: str) -> tuple[str, List[int]]:
    src = str(text or "")
    out: List[str] = []
    mapping: List[int] = []
    prev_space = False
    for i, ch in enumerate(src):
        c = ch
        if c in "{}[]()$`":
            continue
        if c == "\\":
            # Ignore LaTeX backslash markers to improve robustness.
            continue
        if c.isspace():
            if out and (not prev_space):
                out.append(" ")
                mapping.append(i)
                prev_space = True
            continue
        out.append(c.lower())
        mapping.append(i)
        prev_space = False

    # Trim leading/trailing spaces while preserving index mapping.
    while out and out[0] == " ":
        out.pop(0)
        mapping.pop(0)
    while out and out[-1] == " ":
        out.pop()
        mapping.pop()
    return "".join(out), mapping


def _locate_quote_span(text: str, quote: str) -> Dict[str, Any]:
    src = str(text or "")
    q = str(quote or "").strip()
    if not src or not q:
        return {
            "start_char": -1,
            "end_char": -1,
            "line_index": -1,
            "span_valid": False,
            "span_source": "missing_quote",
            "span_ambiguous": False,
        }

    def _pack(start: int, end: int, source: str, ambiguous: bool) -> Dict[str, Any]:
        return {
            "start_char": int(start),
            "end_char": int(end),
            "line_index": int(src.count("\n", 0, start) + 1),
            "span_valid": True,
            "span_source": source,
            "span_ambiguous": bool(ambiguous),
        }

    exact = list(re.finditer(re.escape(q), src))
    if exact:
        m0 = exact[0]
        return _pack(m0.start(), m0.end(), "exact", len(exact) > 1)

    ci = list(re.finditer(re.escape(q), src, flags=re.I))
    if ci:
        m0 = ci[0]
        return _pack(m0.start(), m0.end(), "case_insensitive", len(ci) > 1)

    parts = [re.escape(x) for x in re.split(r"\s+", q) if x]
    if parts:
        pat = r"\s+".join(parts)
        ws = list(re.finditer(pat, src, flags=re.I))
        if ws:
            m0 = ws[0]
            return _pack(m0.start(), m0.end(), "whitespace_fuzzy", len(ws) > 1)

    src_norm, src_map = _collapse_text_for_match(src)
    q_norm, _ = _collapse_text_for_match(q)
    if src_norm and q_norm:
        k = src_norm.find(q_norm)
        if k >= 0:
            s = src_map[k]
            e = src_map[min(len(src_map) - 1, k + len(q_norm) - 1)] + 1
            return _pack(s, e, "normalized_substring", False)

    return {
        "start_char": -1,
        "end_char": -1,
        "line_index": -1,
        "span_valid": False,
        "span_source": "not_found",
        "span_ambiguous": False,
    }


def _normalize_concrete_error(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _max_reached(max_errors: int, count: int) -> bool:
    return max_errors > 0 and count >= max_errors


def _parse_paragraph_index(value: Any) -> int:
    idx = _safe_int(value)
    if idx is None:
        return -1
    return int(idx) if int(idx) >= 1 else -1


def _normalize_error_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    err = _normalize_concrete_error(item.get("error"))
    if not err:
        return None
    quote = _normalize_concrete_error(item.get("quote"))
    return {
        "error": err,
        "quote": quote,
        "paragraph_index": _parse_paragraph_index(item.get("paragraph_index")),
    }


def _dedupe_error_items(items: List[Dict[str, Any]], max_errors: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        norm = _normalize_error_item(raw)
        if not norm:
            continue
        key = (
            str(norm.get("error") or "").casefold(),
            str(norm.get("quote") or "").casefold(),
            int(norm.get("paragraph_index") or -1),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(norm)
        if _max_reached(max_errors, len(out)):
            break
    return out


def _error_token_set(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(t) >= 3}


def _looks_reference_artifact(error_text: str) -> bool:
    s = str(error_text or "").strip().lower()
    patterns = [
        "reference answer",
        "uses a different variable",
        "not defined in the student's answer",
        "placeholder for",
        "reference uses",
    ]
    return any(p in s for p in patterns)


def _filter_error_items_for_quality(items: List[Dict[str, Any]], max_errors: int) -> List[Dict[str, Any]]:
    deduped = _dedupe_error_items(items, 0)
    out: List[Dict[str, Any]] = []
    quote_counter: Dict[str, int] = {}

    for item in deduped:
        err = str(item.get("error") or "")
        if _looks_reference_artifact(err):
            continue

        lower_err = err.lower()
        low_conf_markers = [
            "not an error",
            "minor omission",
            "might be",
            "could be",
            "uncertain",
            "not necessarily",
        ]
        if any(m in lower_err for m in low_conf_markers):
            continue

        tok = _error_token_set(err)
        if len(tok) < 5:
            continue

        quote_key = str(item.get("quote") or "").strip().casefold()
        para_idx = int(item.get("paragraph_index") or -1)

        is_near_dup = False
        for kept in out:
            same_para = int(kept.get("paragraph_index") or -1) == para_idx and para_idx >= 1
            same_quote = str(kept.get("quote") or "").strip().casefold() == quote_key and bool(quote_key)
            if not same_para and not same_quote:
                continue
            ktok = _error_token_set(str(kept.get("error") or ""))
            if not tok or not ktok:
                continue
            jac = len(tok & ktok) / max(1, len(tok | ktok))
            if jac >= 0.72:
                is_near_dup = True
                break
        if is_near_dup:
            continue

        if quote_key:
            c = quote_counter.get(quote_key, 0)
            if c >= 1:
                continue
            quote_counter[quote_key] = c + 1

        out.append(item)
        if _max_reached(max_errors, len(out)):
            break

    return out


def _extract_error_items_from_free_text(text: str, max_errors: int) -> List[Dict[str, Any]]:
    src = str(text or "").strip()
    if not src:
        return []

    lines = [re.sub(r"^[-*\d\.)\s]+", "", ln).strip() for ln in src.splitlines()]
    lines = [ln for ln in lines if ln]

    out: List[Dict[str, Any]] = []
    for ln in lines:
        if len(ln) < 12:
            continue
        norm = _normalize_concrete_error(ln)
        if norm:
            out.append({"error": norm, "quote": "", "paragraph_index": -1})
        if _max_reached(max_errors, len(out)):
            break
    return out


def _extract_error_items_by_regex(text: str, max_errors: int) -> List[Dict[str, Any]]:
    src = str(text or "")
    if not src:
        return []

    out: List[Dict[str, Any]] = []
    obj_iter = list(re.finditer(r"\{[^{}]*\}", src, flags=re.S))
    for m_obj in obj_iter:
        block = m_obj.group(0)
        m_err = re.search(r'"error"\s*:\s*"((?:\\.|[^"\\])*)"', block)
        if not m_err:
            continue
        m_quote = re.search(r'"quote"\s*:\s*"((?:\\.|[^"\\])*)"', block)
        m_para = re.search(r'"paragraph_index"\s*:\s*([0-9]+)', block)
        try:
            decoded_error = json.loads(f'"{m_err.group(1)}"')
        except Exception:
            decoded_error = m_err.group(1)
        decoded_quote = ""
        if m_quote:
            try:
                decoded_quote = json.loads(f'"{m_quote.group(1)}"')
            except Exception:
                decoded_quote = m_quote.group(1)

        norm = _normalize_concrete_error(decoded_error)
        quote = _normalize_concrete_error(decoded_quote)
        if norm:
            out.append(
                {
                    "error": norm,
                    "quote": quote,
                    "paragraph_index": _parse_paragraph_index(m_para.group(1) if m_para else None),
                }
            )
        if _max_reached(max_errors, len(out)):
            break

    if out:
        return _dedupe_error_items(out, max_errors)

    # Fallback: extract error values even when quote is absent/malformed.
    matches = re.findall(r'"error"\s*:\s*"((?:\\.|[^"\\])*)"', src)
    for m in matches:
        try:
            decoded_error = json.loads(f'"{m}"')
        except Exception:
            decoded_error = m
        norm = _normalize_concrete_error(decoded_error)
        if norm:
            out.append({"error": norm, "quote": "", "paragraph_index": -1})
        if _max_reached(max_errors, len(out)):
            break
    return _dedupe_error_items(out, max_errors)


def _extract_error_items_from_payload(payload: Dict[str, Any], max_errors: int) -> List[Dict[str, Any]]:
    raw_errors = None
    if isinstance(payload, dict):
        raw_errors = payload.get("errors")
        if raw_errors is None and isinstance(payload.get("data"), list):
            raw_errors = payload.get("data")
    if not isinstance(raw_errors, list):
        return []

    out: List[Dict[str, Any]] = []
    for item in raw_errors:
        quote = ""
        paragraph_index = -1
        if isinstance(item, dict):
            candidate = str(item.get("error") or "").strip()
            quote = str(item.get("quote") or item.get("evidence_quote") or item.get("span_quote") or "").strip()
            paragraph_index = _parse_paragraph_index(item.get("paragraph_index"))
        else:
            candidate = str(item or "").strip()

        if not candidate:
            continue
        normalized = _normalize_concrete_error(candidate)
        quote_norm = _normalize_concrete_error(quote)
        if normalized:
            out.append({"error": normalized, "quote": quote_norm, "paragraph_index": paragraph_index})
        if _max_reached(max_errors, len(out)):
            break
    return _dedupe_error_items(out, max_errors)


def _paragraph_ranges(text: str) -> List[Dict[str, Any]]:
    src = str(text or "")
    if not src:
        return []

    # Build compact location bins from punctuation/length, independent of original blank-line paragraphs.
    target_len = 220
    min_len = 120
    max_len = 360
    n = len(src)
    boundary_set = {0, n}
    for m in re.finditer(r"[。！？!?；;](?:\s+|$)|\n+", src):
        boundary_set.add(m.end())
    boundaries = sorted(boundary_set)

    out: List[Dict[str, Any]] = []
    start = 0
    para_idx = 0
    while start < n:
        if n - start <= max_len:
            end = n
        else:
            low = min(n, start + min_len)
            high = min(n, start + max_len)
            desired = min(n, start + target_len)
            candidates = [b for b in boundaries if low <= b <= high]
            if candidates:
                end = min(candidates, key=lambda b: abs(b - desired))
            else:
                end = high

        s = start
        e = max(start, end)
        while s < e and src[s].isspace():
            s += 1
        while e > s and src[e - 1].isspace():
            e -= 1

        if e > s:
            para_idx += 1
            out.append(
                {
                    "paragraph_index": para_idx,
                    "start_char": s,
                    "end_char": e,
                    "text": src[s:e],
                }
            )
        start = end if end > start else start + 1

    if not out and src.strip():
        out.append({"paragraph_index": 1, "start_char": 0, "end_char": len(src), "text": src})
    return out


def _expand_span_to_context_window(
    text: str,
    start_char: int,
    end_char: int,
    *,
    left_context: int = 90,
    right_context: int = 120,
    max_window: int = 320,
) -> Dict[str, int]:
    src = str(text or "")
    n = len(src)
    if n <= 0 or start_char < 0 or end_char <= start_char:
        return {"start_char": -1, "end_char": -1}

    s = max(0, int(start_char) - left_context)
    e = min(n, int(end_char) + right_context)

    # Snap to token boundaries for readability/stability.
    while s > 0 and (not src[s - 1].isspace()) and (int(start_char) - s) < (left_context + 50):
        s -= 1
    while e < n and (not src[e].isspace()) and (e - int(end_char)) < (right_context + 50):
        e += 1

    if e - s > max_window:
        mid = (int(start_char) + int(end_char)) // 2
        half = max_window // 2
        s = max(0, mid - half)
        e = min(n, s + max_window)

    while s < e and src[s].isspace():
        s += 1
    while e > s and src[e - 1].isspace():
        e -= 1
    return {"start_char": s if e > s else -1, "end_char": e if e > s else -1}


def _paragraph_from_offset(paragraphs: List[Dict[str, Any]], offset: int) -> Optional[Dict[str, Any]]:
    if offset < 0:
        return None
    for p in paragraphs:
        s = int(p.get("start_char") or -1)
        e = int(p.get("end_char") or -1)
        if s <= offset < e:
            return p
    return None


def _paragraph_by_index(paragraphs: List[Dict[str, Any]], paragraph_index: int) -> Optional[Dict[str, Any]]:
    if paragraph_index <= 0:
        return None
    for p in paragraphs:
        if int(p.get("paragraph_index") or -1) == paragraph_index:
            return p
    return None


def _infer_paragraph_location(
    *,
    prediction: str,
    error_text: str,
    quote: str,
    span: Dict[str, Any],
    declared_paragraph_index: int,
) -> Dict[str, Any]:
    paragraphs = _paragraph_ranges(prediction)
    if not paragraphs:
        return {
            "paragraph_index": -1,
            "paragraph_start_char": -1,
            "paragraph_end_char": -1,
            "paragraph_valid": False,
            "paragraph_source": "missing_text",
        }

    p_declared = _paragraph_by_index(paragraphs, declared_paragraph_index)
    if p_declared is not None:
        return {
            "paragraph_index": int(p_declared.get("paragraph_index") or -1),
            "paragraph_start_char": int(p_declared.get("start_char") or -1),
            "paragraph_end_char": int(p_declared.get("end_char") or -1),
            "paragraph_valid": True,
            "paragraph_source": "model_declared",
        }

    if bool(span.get("span_valid")):
        p_span = _paragraph_from_offset(paragraphs, int(span.get("start_char") or -1))
        if p_span is not None:
            ctx = _expand_span_to_context_window(
                prediction,
                int(span.get("start_char") or -1),
                int(span.get("end_char") or -1),
            )
            return {
                "paragraph_index": int(p_span.get("paragraph_index") or -1),
                "paragraph_start_char": int(ctx.get("start_char") or p_span.get("start_char") or -1),
                "paragraph_end_char": int(ctx.get("end_char") or p_span.get("end_char") or -1),
                "paragraph_valid": True,
                "paragraph_source": "from_span_context",
            }

    if quote:
        q_span = _locate_quote_span(prediction, quote)
        if bool(q_span.get("span_valid")):
            p_quote = _paragraph_from_offset(paragraphs, int(q_span.get("start_char") or -1))
            if p_quote is not None:
                ctx = _expand_span_to_context_window(
                    prediction,
                    int(q_span.get("start_char") or -1),
                    int(q_span.get("end_char") or -1),
                )
                return {
                    "paragraph_index": int(p_quote.get("paragraph_index") or -1),
                    "paragraph_start_char": int(ctx.get("start_char") or p_quote.get("start_char") or -1),
                    "paragraph_end_char": int(ctx.get("end_char") or p_quote.get("end_char") or -1),
                    "paragraph_valid": True,
                    "paragraph_source": "from_quote_context",
                }

    err_tokens = {t for t in re.findall(r"[a-zA-Z0-9_]+", str(error_text or "").lower()) if len(t) >= 3}
    if err_tokens:
        best = None
        best_overlap = 0
        for p in paragraphs:
            p_tokens = {t for t in re.findall(r"[a-zA-Z0-9_]+", str(p.get("text") or "").lower()) if len(t) >= 3}
            overlap = len(err_tokens & p_tokens)
            if overlap > best_overlap:
                best_overlap = overlap
                best = p
        if best is not None and best_overlap > 0:
            return {
                "paragraph_index": int(best.get("paragraph_index") or -1),
                "paragraph_start_char": int(best.get("start_char") or -1),
                "paragraph_end_char": int(best.get("end_char") or -1),
                "paragraph_valid": True,
                "paragraph_source": "token_overlap",
            }

    return {
        "paragraph_index": -1,
        "paragraph_start_char": -1,
        "paragraph_end_char": -1,
        "paragraph_valid": False,
        "paragraph_source": "not_found",
    }


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
        "error_items": [],
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

    para_count = len(_paragraph_ranges(prediction))
    if max_errors > 0:
        error_count_instruction = f"Output up to {max_errors} errors."
        extraction_cap = max_errors
    else:
        error_count_instruction = (
            "Output all identifiable errors exhaustively. "
            "Do not impose an arbitrary limit unless two items are genuinely duplicates."
        )
        extraction_cap = 0

    system_prompt = (
        "You are a strict physics evaluator for building a recall benchmark. "
        "Extract concrete mistakes from the student answer itself, not abstract rule templates. "
        "Each error must point to a specific wrong statement/formula in the student's answer. "
        "Use English only. No rubric references."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Student answer:\n{prediction}\n\n"
        f"Reference answer:\n{answer}\n\n"
        "Return JSON only with this schema:\n"
        "{\n"
        "  \"errors\": [\n"
        "    {\"error\": \"Describe the concrete mistake\", \"quote\": \"Exact substring copied from Student answer\", \"paragraph_index\": 1}\n"
        "  ]\n"
        "}\n\n"
        "Style requirements for each item:\n"
        "1) Focus on specific wrong content in the answer (wrong formula, wrong sign, wrong substitution, contradiction, unjustified step).\n"
        "2) quote must be an exact copied substring from Student answer so it can be located by string match.\n"
        "3) Keep one concrete error per item, and avoid overlaps between items.\n"
        "4) Do NOT use generic templates like 'should ... but ...'.\n"
        "5) Do NOT output text not present in student answer into quote.\n"
        "6) paragraph_index is 1-based interval index in the student answer (intervals are auto-chunked by the system, not original blank-line paragraphs); if uncertain use -1.\n"
        f"Student answer has approximately {para_count} intervals.\n\n"
        f"{error_count_instruction}"
    )

    def _repair_raw_output(raw_text: str) -> List[Dict[str, Any]]:
        raw_text = str(raw_text or "").strip()
        if not raw_text:
            return []
        repair_prompt = (
            "The previous model output may be truncated or malformed JSON. "
            "Rewrite it into valid JSON only with schema: {\"errors\":[{\"error\":\"...\",\"quote\":\"...\",\"paragraph_index\":1}]}. "
            "Keep only complete, meaningful, concrete mistakes grounded in the student answer. "
            "quote must be an exact substring from the student answer. "
            "Use paragraph_index as 1-based index in student answer, or -1 if uncertain.\n"
            f"{error_count_instruction}\n\n"
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
                repaired_items = _extract_error_items_from_payload(repaired_payload, max_errors=extraction_cap)
                if repaired_items:
                    return repaired_items
            return _extract_error_items_by_regex(repaired_raw, max_errors=extraction_cap)
        except Exception as e:
            result["attempt_logs"].append({"repair_exception": f"{type(e).__name__}: {str(e)[:300]}"})
            return []

    def _self_check_more(existing_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not existing_items:
            return []
        review_prompt = (
            "You produced an initial list of physics errors. "
            "Find additional missed concrete errors from the student answer only. "
            "Do NOT repeat existing items. Return JSON with schema: "
            "{\"errors\":[{\"error\":\"...\",\"quote\":\"...\",\"paragraph_index\":1}]}. "
            f"{error_count_instruction}\n\n"
            f"Question:\n{question}\n\n"
            f"Student answer:\n{prediction}\n\n"
            f"Reference answer:\n{answer}\n\n"
            "Existing errors:\n"
            + json.dumps(existing_items, ensure_ascii=False)
        )
        try:
            resp3 = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You expand an error list with only new, non-duplicate items."},
                    {"role": "user", "content": review_prompt},
                ],
                temperature=0.0,
                max_tokens=1200,
                response_format={"type": "json_object"},
            )
            review_raw = (resp3.choices[0].message.content or "").strip()
            result["raw_outputs"].append(review_raw)
            payload = _extract_json_object(review_raw)
            if payload:
                return _extract_error_items_from_payload(payload, max_errors=extraction_cap)
            return _extract_error_items_by_regex(review_raw, max_errors=extraction_cap)
        except Exception as e:
            result["attempt_logs"].append({"self_check_exception": f"{type(e).__name__}: {str(e)[:300]}"})
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
                "max_tokens": 1800,
            }
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            try:
                resp = client.chat.completions.create(**kwargs)
                raw = (resp.choices[0].message.content or "").strip()
                result["raw_outputs"].append(raw)

                payload = _extract_json_object(raw)
                error_items: List[Dict[str, Any]] = []
                if payload:
                    error_items = _extract_error_items_from_payload(payload, max_errors=extraction_cap)
                if not error_items:
                    error_items = _extract_error_items_by_regex(raw, max_errors=extraction_cap)
                if not error_items:
                    error_items = _extract_error_items_from_free_text(raw, max_errors=extraction_cap)
                if not error_items and raw:
                    error_items = _repair_raw_output(raw)

                if error_items and len(error_items) <= 8 and not _max_reached(extraction_cap, len(error_items)):
                    extra_items = _self_check_more(error_items)
                    error_items = _dedupe_error_items(error_items + extra_items, extraction_cap)

                error_items = _filter_error_items_for_quality(error_items, extraction_cap)

                errors = [str(it.get("error") or "") for it in error_items if str(it.get("error") or "").strip()]

                result["attempt_logs"].append(
                    {
                        "attempt": attempt + 1,
                        "mode": "json_object" if use_json_mode else "plain",
                        "raw_len": len(raw),
                        "payload_parsed": bool(payload),
                        "errors_extracted": len(error_items),
                    }
                )

                if error_items:
                    result["ok"] = True
                    result["errors"] = errors
                    result["error_items"] = error_items
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

    if max_errors > 0:
        return out[:max_errors]
    return out


def _build_location_gt_entries(sample_id: str, prediction: str, error_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(error_items):
        err = _normalize_concrete_error(item.get("error"))
        quote = _normalize_concrete_error(item.get("quote"))
        if not err:
            continue
        span = _locate_quote_span(prediction, quote)
        paragraph = _infer_paragraph_location(
            prediction=prediction,
            error_text=err,
            quote=quote,
            span=span,
            declared_paragraph_index=_parse_paragraph_index(item.get("paragraph_index")),
        )
        span_valid = bool(span.get("span_valid"))
        paragraph_valid = bool(paragraph.get("paragraph_valid"))
        out.append(
            {
                "error_id": f"{sample_id}_e{i + 1}",
                "error_text": err,
                "answer_quote": quote,
                "start_char": int(span.get("start_char", -1)),
                "end_char": int(span.get("end_char", -1)),
                "line_index": int(span.get("line_index", -1)),
                "span_valid": span_valid,
                "span_source": str(span.get("span_source") or ""),
                "span_ambiguous": bool(span.get("span_ambiguous")),
                "paragraph_index": int(paragraph.get("paragraph_index") or -1),
                "paragraph_start_char": int(paragraph.get("paragraph_start_char") or -1),
                "paragraph_end_char": int(paragraph.get("paragraph_end_char") or -1),
                "paragraph_valid": paragraph_valid,
                "paragraph_source": str(paragraph.get("paragraph_source") or ""),
                "locatable_valid": bool(span_valid or paragraph_valid),
            }
        )
    return out


def _build_question_eval_item(row: Dict[str, Any], expected_has_physics_error: bool, split: str) -> Dict[str, Any]:
    return {
        "id": str(row.get("id") or ""),
        "question": str(row.get("question") or ""),
        "context": str(row.get("context") or ""),
        "prediction": str(row.get("prediction") or ""),
        "answer": str(row.get("answer") or ""),
        "expected_has_physics_error": bool(expected_has_physics_error),
        "eval_split": str(split),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build error-level and question-level physics evaluation sets.")
    parser.add_argument("--input", type=str, default="data/physics_rubric_data_1000.json", help="Legacy shared input path (kept for compatibility).")
    parser.add_argument("--recall-input", type=str, default="data/evaluation_sample_1000_expansion.json")
    parser.add_argument("--precision-input", type=str, default="data/physics_rubric_data_1000.json")
    parser.add_argument("--error-output", type=str, default="")
    parser.add_argument("--question-output", type=str, default="")
    parser.add_argument("--recall-output", type=str, default="data/evaluation_recall_20.json")
    parser.add_argument("--precision-output", type=str, default="data/evaluation_precision_20.json")
    parser.add_argument(
        "--unified-output",
        type=str,
        default="",
        help="Deprecated compatibility output. Prefer --question-output for question-level dataset.",
    )
    parser.add_argument("--recall-size", type=int, default=20)
    parser.add_argument("--precision-size", type=int, default=20)
    parser.add_argument("--skip-recall", action="store_true")
    parser.add_argument("--skip-precision", action="store_true")
    parser.add_argument("--seed", type=int, default=20260409)
    parser.add_argument("--strong-model", type=str, default="gemini-3-flash-preview")
    parser.add_argument(
        "--max-errors",
        type=int,
        default=0,
        help="Maximum GT errors to keep per sample. Use 0 for exhaustive mode.",
    )
    parser.add_argument(
        "--min-valid-gt-per-sample",
        type=int,
        default=1,
        help="Minimum number of locatable GT errors required per recall sample.",
    )
    parser.add_argument(
        "--allow-invalid-location-gt",
        action="store_true",
        help="Allow recall samples with insufficient locatable GT errors.",
    )
    parser.add_argument(
        "--max-recall-scan",
        type=int,
        default=0,
        help="Maximum recall candidates to scan. 0 means scan all recall-input samples.",
    )
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

    recall_candidates = list(recall_data)
    rng.shuffle(recall_candidates)
    if args.max_recall_scan > 0:
        recall_candidates = recall_candidates[: min(len(recall_candidates), int(args.max_recall_scan))]

    precision_rows = rng.sample(relaxed_correct_pool, precision_take) if precision_take > 0 else []

    recall_out: List[Dict[str, Any]] = []
    llm_used = 0
    llm_failed = 0
    gt_total_errors = 0
    gt_locatable_errors = 0
    gt_span_locatable_errors = 0
    gt_paragraph_locatable_errors = 0
    samples_with_zero_valid_gt = 0
    ambiguous_span_count = 0
    recall_candidates_scanned = 0
    recall_rejected_invalid_location = 0
    recall_rejected_non_dict = 0
    recall_rejected_no_errors = 0
    failure_reason_counter: Dict[str, int] = {}
    if not args.skip_recall:
        min_valid_gt = max(0, int(args.min_valid_gt_per_sample))
        enforce_valid_gt = not bool(args.allow_invalid_location_gt)

        for i, row in enumerate(recall_candidates):
            if len(recall_out) >= recall_take:
                break
            if not isinstance(row, dict):
                recall_rejected_non_dict += 1
                continue

            recall_candidates_scanned += 1
            base = _build_base_eval_item(row, fallback_id=f"recall_scan_{i}")
            gen = _strong_model_generation(
                model=args.strong_model,
                question=base["question"],
                prediction=base["prediction"],
                answer=base["answer"],
                max_errors=args.max_errors,
            )
            error_items = gen.get("error_items") if isinstance(gen.get("error_items"), list) else []
            errors = [str(it.get("error") or "") for it in error_items if isinstance(it, dict) and str(it.get("error") or "").strip()]
            location_gt = _build_location_gt_entries(base["id"], base["prediction"], [it for it in error_items if isinstance(it, dict)])
            valid_location_gt = [x for x in location_gt if bool(x.get("locatable_valid"))]

            if gen.get("ok") and errors:
                source = "strong_model"
                llm_used += 1
            else:
                source = "strong_model_failed"
                llm_failed += 1
                reason = str(gen.get("last_error") or "unknown_failure")
                failure_reason_counter[reason] = failure_reason_counter.get(reason, 0) + 1

            if not errors:
                recall_rejected_no_errors += 1
                if enforce_valid_gt:
                    if len(valid_location_gt) < min_valid_gt:
                        recall_rejected_invalid_location += 1
                    continue

            if enforce_valid_gt and len(valid_location_gt) < min_valid_gt:
                recall_rejected_invalid_location += 1
                continue

            gt_total_errors += len(location_gt)
            gt_locatable_errors += len(valid_location_gt)
            gt_span_locatable_errors += len([x for x in location_gt if bool(x.get("span_valid"))])
            gt_paragraph_locatable_errors += len([x for x in location_gt if bool(x.get("paragraph_valid"))])
            ambiguous_span_count += len([x for x in location_gt if bool(x.get("span_ambiguous"))])
            if location_gt and not valid_location_gt:
                samples_with_zero_valid_gt += 1

            recall_out.append(
                {
                    **base,
                    "physics_error_examples": [{"error": e} for e in errors],
                    "physics_error_gt": location_gt,
                    "physics_error_gt_valid_count": len(valid_location_gt),
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

    error_output_path = str(args.error_output or "").strip() or str(args.recall_output)
    question_output_path = str(args.question_output or "").strip()

    recall_path = Path(error_output_path)
    precision_path = Path(args.precision_output)
    recall_path.parent.mkdir(parents=True, exist_ok=True)
    precision_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_recall:
        recall_path.write_text(json.dumps(recall_out, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.skip_precision:
        precision_path.write_text(json.dumps(precision_out, ensure_ascii=False, indent=2), encoding="utf-8")

    question_out: List[Dict[str, Any]] = []
    for x in recall_out:
        if isinstance(x, dict):
            question_out.append(_build_question_eval_item(x, expected_has_physics_error=True, split="wrong"))
    for x in precision_out:
        if isinstance(x, dict):
            question_out.append(_build_question_eval_item(x, expected_has_physics_error=False, split="right"))

    if question_output_path:
        qpath = Path(question_output_path)
        qpath.parent.mkdir(parents=True, exist_ok=True)
        qpath.write_text(json.dumps(question_out, ensure_ascii=False, indent=2), encoding="utf-8")

    unified_output_path = str(args.unified_output or "").strip()
    if unified_output_path:
        unified_rows: List[Dict[str, Any]] = list(question_out)
        up = Path(unified_output_path)
        up.parent.mkdir(parents=True, exist_ok=True)
        up.write_text(json.dumps(unified_rows, ensure_ascii=False, indent=2), encoding="utf-8")

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
        "recall_target_size": recall_take if not args.skip_recall else 0,
        "recall_collected_size": len(recall_out) if not args.skip_recall else 0,
        "recall_shortfall": (recall_take - len(recall_out)) if not args.skip_recall else 0,
        "recall_candidates_scanned": recall_candidates_scanned,
        "recall_rejected_invalid_location": recall_rejected_invalid_location,
        "recall_rejected_no_errors": recall_rejected_no_errors,
        "recall_rejected_non_dict": recall_rejected_non_dict,
        "min_valid_gt_per_sample": int(args.min_valid_gt_per_sample),
        "allow_invalid_location_gt": bool(args.allow_invalid_location_gt),
        "gt_total_errors": gt_total_errors,
        "gt_locatable_errors": gt_locatable_errors,
        "gt_span_locatable_errors": gt_span_locatable_errors,
        "gt_paragraph_locatable_errors": gt_paragraph_locatable_errors,
        "gt_location_valid_ratio": (gt_locatable_errors / gt_total_errors) if gt_total_errors else 0.0,
        "samples_with_zero_valid_gt": samples_with_zero_valid_gt,
        "ambiguous_span_count": ambiguous_span_count,
        "error_output": str(recall_path) if not args.skip_recall else None,
        "question_output": question_output_path or None,
        "question_size": len(question_out),
        "recall_output": str(recall_path) if not args.skip_recall else None,
        "precision_output": str(precision_path) if not args.skip_precision else None,
        "unified_output": unified_output_path or None,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
