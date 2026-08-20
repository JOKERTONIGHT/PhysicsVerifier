from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from dotenv import load_dotenv

# Reuse location-based eval helpers for consistent finding extraction.
from evaluate_physics_eval_sets import (  # type: ignore
    _collect_pred_findings,
    _extract_gt_entries,
    _fill_missing_pred_locations,
    _index_by_id,
    _load_json,
    _match_by_location,
)


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
        raise SystemExit("OPENAI_API_KEY is required for semantic upper-bound evaluation.")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    return openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)


def _llm_semantic_coverage(
    client: Any,
    model: str,
    *,
    question: str,
    gt_errors: List[str],
    findings_text: List[str],
    timeout: float,
) -> Optional[Dict[str, Any]]:
    if not gt_errors:
        return {"covered": [], "reasons": []}

    system_prompt = (
        "You are a physics-error recall evaluator. "
        "Decide whether each ground-truth (GT) error is semantically covered by at least one predicted finding. "
        "Coverage means the prediction identifies the same concrete mistake, wrong formula, wrong assumption, "
        "or contradiction — even if wording differs. "
        "Do NOT require exact quote overlap. "
        "If findings are empty, all GT errors are uncovered. "
        "Return JSON only: "
        '{"covered":[true/false,...],"reasons":["short reason per GT item"]} '
        "with arrays the same length as GT list."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        "Ground-truth errors:\n"
        + json.dumps(gt_errors, ensure_ascii=False)
        + "\n\nPredicted findings:\n"
        + json.dumps(findings_text, ensure_ascii=False)
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=1200,
            response_format={"type": "json_object"},
            timeout=timeout,
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        data = _extract_json_object(raw)
        if not isinstance(data, dict):
            return {"error": f"invalid_json: {raw[:300]}"}
        covered = data.get("covered") if isinstance(data, dict) else None
        reasons = data.get("reasons") if isinstance(data, dict) else None
        if not isinstance(covered, list):
            return None
        out_cov: List[bool] = []
        out_reasons: List[str] = []
        for i in range(len(gt_errors)):
            out_cov.append(bool(covered[i]) if i < len(covered) else False)
            if isinstance(reasons, list) and i < len(reasons):
                out_reasons.append(str(reasons[i] or ""))
            else:
                out_reasons.append("")
        return {"covered": out_cov, "reasons": out_reasons}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:300]}"}


def _finding_texts(findings: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for f in findings:
        msg = str(f.get("message") or f.get("text") or "").strip()
        quote = str(f.get("quote") or "").strip()
        if msg and quote:
            out.append(f"{msg} | quote: {quote}")
        elif msg:
            out.append(msg)
        elif quote:
            out.append(f"quote: {quote}")
    return out


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Estimate semantic recall upper bound by judging GT coverage with a strong LLM."
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--audit", type=str, default="")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--flush-every", type=int, default=5)
    parser.add_argument(
        "--location-iou-threshold",
        type=float,
        default=0.5,
        help="Also compute location recall for comparison.",
    )
    parser.add_argument("--location-coverage-threshold", type=float, default=0.6)
    args = parser.parse_args()

    ds = _load_json(args.dataset)
    pred = _load_json(args.results)
    audit = _load_json(args.audit) if str(args.audit or "").strip() else []

    if not isinstance(ds, list):
        raise SystemExit("Dataset must be a JSON array.")

    pred_idx = _index_by_id(pred if isinstance(pred, list) else [])
    audit_idx = _index_by_id(audit if isinstance(audit, list) else [])
    client = _openai_client()

    rows = [r for r in ds if isinstance(r, dict) and str(r.get("id") or "").strip()]
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]

    detail_rows: List[Dict[str, Any]] = []
    total_gt = 0
    loc_matched = 0
    sem_matched = 0
    sem_only_matched = 0
    loc_only_matched = 0
    api_failures = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows):
        sid = str(row.get("id") or "").strip()
        gt_entries = _extract_gt_entries(row, sid)
        gt_texts = [str(g.get("error_text") or "") for g in gt_entries]
        loc_gt_entries = [x for x in gt_entries if bool(x.get("locatable_valid"))]

        pred_item = pred_idx.get(sid, {})
        audit_item = audit_idx.get(sid, {})
        findings = _collect_pred_findings(pred_item, audit_item)
        findings = _fill_missing_pred_locations(findings, answer_text=str(row.get("prediction") or ""))
        finding_texts = _finding_texts(findings)

        matched_loc_ids: Set[str] = set()
        if loc_gt_entries:
            matched_loc_ids, _, _ = _match_by_location(
                loc_gt_entries,
                findings,
                iou_threshold=float(args.location_iou_threshold),
                coverage_threshold=float(args.location_coverage_threshold),
            )

        sem = _llm_semantic_coverage(
            client,
            str(args.model),
            question=str(row.get("question") or ""),
            gt_errors=gt_texts,
            findings_text=finding_texts,
            timeout=float(args.timeout),
        )
        if not sem or sem.get("error"):
            api_failures += 1
            covered_flags = [False] * len(gt_texts)
            sem_reasons: List[str] = []
        else:
            covered_flags = [bool(x) for x in (sem.get("covered") or [])]
            sem_reasons = [str(x) for x in (sem.get("reasons") or [])]

        per_gt: List[Dict[str, Any]] = []
        sample_loc = 0
        sample_sem = 0
        for i, g in enumerate(gt_entries):
            gid = str(g.get("error_id") or f"{sid}_e{i+1}")
            loc_hit = gid in matched_loc_ids
            sem_hit = bool(covered_flags[i]) if i < len(covered_flags) else False
            if loc_hit:
                sample_loc += 1
            if sem_hit:
                sample_sem += 1
            if sem_hit and not loc_hit:
                sem_only_matched += 1
            if loc_hit and not sem_hit:
                loc_only_matched += 1
            per_gt.append(
                {
                    "error_id": gid,
                    "error_text": str(g.get("error_text") or ""),
                    "location_matched": loc_hit,
                    "semantic_matched": sem_hit,
                    "semantic_reason": sem_reasons[i] if i < len(sem_reasons) else "",
                }
            )

        total_gt += len(gt_entries)
        loc_matched += sample_loc
        sem_matched += sample_sem

        detail_rows.append(
            {
                "id": sid,
                "gt_error_count": len(gt_entries),
                "pred_finding_count": len(findings),
                "location_matched_count": sample_loc,
                "semantic_matched_count": sample_sem,
                "sample_location_recall": (sample_loc / len(gt_entries)) if gt_entries else 0.0,
                "sample_semantic_recall": (sample_sem / len(gt_entries)) if gt_entries else 0.0,
                "findings_preview": finding_texts[:5],
                "per_gt": per_gt,
            }
        )

        if (idx + 1) % max(1, int(args.flush_every)) == 0:
            partial = {
                "summary": {
                    "processed_samples": len(detail_rows),
                    "semantic_recall": (sem_matched / total_gt) if total_gt else 0.0,
                    "location_recall": (loc_matched / total_gt) if total_gt else 0.0,
                },
                "details": detail_rows,
            }
            out_path.write_text(json.dumps(partial, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[progress] {idx + 1}/{len(rows)} semantic_recall={partial['summary']['semantic_recall']:.3f}")

        time.sleep(0.2)

    output = {
        "summary": {
            "model": str(args.model),
            "dataset_size": len(detail_rows),
            "total_gt_errors": total_gt,
            "location_matched_gt_errors": loc_matched,
            "semantic_matched_gt_errors": sem_matched,
            "location_recall": (loc_matched / total_gt) if total_gt else 0.0,
            "semantic_recall_upper_bound": (sem_matched / total_gt) if total_gt else 0.0,
            "semantic_minus_location_gain": ((sem_matched - loc_matched) / total_gt) if total_gt else 0.0,
            "semantic_only_matches": sem_only_matched,
            "location_only_matches": loc_only_matched,
            "api_failures": api_failures,
        },
        "details": detail_rows,
    }
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
