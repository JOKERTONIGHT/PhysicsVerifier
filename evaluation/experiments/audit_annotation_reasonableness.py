from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def _openai_client():
    import openai  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required.")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    return openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)


def _audit_gt_batch(
    client: Any,
    model: str,
    *,
    question: str,
    prediction: str,
    answer: str,
    gt_items: List[Dict[str, Any]],
    timeout: float,
) -> Dict[str, Any]:
    payload = [
        {
            "error_id": str(g.get("error_id") or ""),
            "error_text": str(g.get("error_text") or ""),
            "answer_quote": str(g.get("answer_quote") or ""),
        }
        for g in gt_items
    ]
    system_prompt = (
        "You audit physics-error benchmark annotations. "
        "For each GT item, judge whether it is a valid concrete physics mistake in the student answer. "
        "Labels: valid, over_granular, duplicate, questionable, not_error. "
        "Also judge quote_grounded: true if answer_quote is substantively present in student answer. "
        "Return JSON only: {\"items\":[{\"error_id\":\"...\",\"label\":\"valid\",\"quote_grounded\":true,\"reason\":\"...\"}]}"
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Reference answer:\n{answer}\n\n"
        f"Student answer:\n{prediction}\n\n"
        f"GT items:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=2500,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        data = _extract_json_object(resp.choices[0].message.content or "")
        items = data.get("items") if isinstance(data, dict) else None
        if not isinstance(items, list):
            return {"ok": False, "error": "invalid_response"}
        return {"ok": True, "items": items}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"}


def _audit_sample(
    client: Any,
    model: str,
    *,
    question: str,
    prediction: str,
    answer: str,
    gt_items: List[Dict[str, Any]],
    timeout: float,
    batch_gt_size: int,
) -> Dict[str, Any]:
    if not gt_items:
        return {"ok": True, "items": []}
    chunk = max(1, int(batch_gt_size))
    merged_items: List[Dict[str, Any]] = []
    last_error: Optional[str] = None
    for i in range(0, len(gt_items), chunk):
        part = gt_items[i : i + chunk]
        audit = _audit_gt_batch(
            client,
            model,
            question=question,
            prediction=prediction,
            answer=answer,
            gt_items=part,
            timeout=timeout,
        )
        if not audit.get("ok"):
            last_error = str(audit.get("error") or "invalid_response")
            continue
        for item in audit.get("items") or []:
            if isinstance(item, dict):
                merged_items.append(item)
    if merged_items:
        return {"ok": True, "items": merged_items}
    return {"ok": False, "error": last_error or "invalid_response"}


def _load_sample_ids(path: str) -> List[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(x).strip() for x in payload if str(x).strip()]
    raise SystemExit("--sample-ids must point to a JSON array of sample ids.")


def _merge_audit_details(existing: List[Dict[str, Any]], new_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(d.get("id") or ""): d for d in existing if str(d.get("id") or "")}
    for row in new_rows:
        sid = str(row.get("id") or "")
        if sid:
            by_id[sid] = row
    return [by_id[sid] for sid in sorted(by_id.keys())]


def _summarize_details(details: List[Dict[str, Any]], model: str) -> Dict[str, Any]:
    label_counter: Dict[str, int] = {}
    quote_grounded = 0
    total_items = 0
    for row in details:
        if not row.get("audit_ok"):
            continue
        for item in row.get("items") or []:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or "unknown").strip().lower()
            label_counter[label] = label_counter.get(label, 0) + 1
            total_items += 1
            if bool(item.get("quote_grounded")):
                quote_grounded += 1
    valid = label_counter.get("valid", 0)
    questionable = sum(label_counter.get(k, 0) for k in ("questionable", "not_error", "duplicate", "over_granular"))
    return {
        "model": str(model),
        "samples": len(details),
        "total_gt_items": total_items,
        "valid_count": valid,
        "valid_ratio": (valid / total_items) if total_items else 0.0,
        "questionable_or_invalid_ratio": (questionable / total_items) if total_items else 0.0,
        "quote_grounded_ratio": (quote_grounded / total_items) if total_items else 0.0,
        "label_counts": label_counter,
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Audit GT annotation reasonableness with a strong LLM.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--sample-ids", type=str, default="", help="Optional JSON array file of sample ids to audit.")
    parser.add_argument("--merge-from", type=str, default="", help="Optional existing audit JSON to merge/replace by sample id.")
    parser.add_argument("--batch-gt-size", type=int, default=4, help="Audit GT items in chunks to avoid oversized responses.")
    args = parser.parse_args()

    rows = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("Dataset must be a JSON array.")
    sample_ids_path = str(args.sample_ids or "").strip()
    if sample_ids_path:
        wanted = set(_load_sample_ids(sample_ids_path))
        rows = [r for r in rows if isinstance(r, dict) and str(r.get("id") or "") in wanted]
    elif int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]

    client = _openai_client()
    details: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "")
        gt_items = row.get("physics_error_gt") if isinstance(row.get("physics_error_gt"), list) else []
        audit = _audit_sample(
            client,
            str(args.model),
            question=str(row.get("question") or ""),
            prediction=str(row.get("prediction") or ""),
            answer=str(row.get("answer") or ""),
            gt_items=[g for g in gt_items if isinstance(g, dict)],
            timeout=float(args.timeout),
            batch_gt_size=int(args.batch_gt_size),
        )
        sample_labels: Dict[str, int] = {}
        item_rows: List[Dict[str, Any]] = []
        if audit.get("ok"):
            for item in audit.get("items") or []:
                if not isinstance(item, dict):
                    continue
                label = str(item.get("label") or "unknown").strip().lower()
                sample_labels[label] = sample_labels.get(label, 0) + 1
                item_rows.append(item)
        details.append({"id": sid, "gt_count": len(gt_items), "audit_ok": bool(audit.get("ok")), "items": item_rows, "error": audit.get("error")})
        print(f"[audit] {sid} labels={sample_labels}", flush=True)
        time.sleep(0.2)

    merge_from = str(args.merge_from or "").strip()
    if merge_from:
        existing = json.loads(Path(merge_from).read_text(encoding="utf-8"))
        prior_details = existing.get("details") if isinstance(existing, dict) else None
        if not isinstance(prior_details, list):
            raise SystemExit("--merge-from must contain a details array.")
        details = _merge_audit_details(prior_details, details)

    out = {
        "summary": _summarize_details(details, str(args.model)),
        "details": details,
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
