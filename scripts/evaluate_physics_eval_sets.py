from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _index_by_id(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in items:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "").strip()
        if sid:
            out[sid] = row
    return out


def _collect_pred_findings(pred_item: Dict[str, Any], audit_item: Optional[Dict[str, Any]]) -> List[str]:
    findings: List[str] = []

    diagnostics = pred_item.get("diagnostics") if isinstance(pred_item, dict) else []
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    for d in diagnostics:
        if not isinstance(d, dict):
            continue
        parts = [str(d.get("rule") or ""), str(d.get("message") or ""), str(d.get("evidence") or "")]
        s = " | ".join([p for p in parts if p.strip()])
        if s.strip():
            findings.append(s)

    if isinstance(audit_item, dict):
        checks = audit_item.get("experience_code_checks")
        checks = checks if isinstance(checks, list) else []
        for c in checks:
            if not isinstance(c, dict):
                continue
            res = str(c.get("result") or "").strip().lower()
            if res != "fail":
                continue
            parts = [str(c.get("rule") or ""), str(c.get("message") or ""), str(c.get("evidence") or "")]
            s = " | ".join([p for p in parts if p.strip()])
            if s.strip():
                findings.append(s)

    # Ordered unique
    out: List[str] = []
    seen = set()
    for f in findings:
        key = f.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(t) >= 3]


def _keyword_match(gt_error: str, findings: List[str], min_overlap: int = 3) -> bool:
    gt_tokens = set(_tokenize(gt_error))
    if not gt_tokens:
        return False
    for f in findings:
        f_tokens = set(_tokenize(f))
        if len(gt_tokens & f_tokens) >= min_overlap:
            return True
    return False


def _llm_match_coverage(model: str, gt_errors: List[str], findings: List[str]) -> Optional[List[bool]]:
    try:
        import os
        import openai  # type: ignore
    except Exception:
        return None

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    client = openai.OpenAI(base_url=base_url)

    system_prompt = (
        "You are an evaluator for physics-error detection recall. "
        "Given ground-truth error statements and predicted findings, decide whether each GT error is covered. "
        "Return JSON only: {\"covered\": [true/false, ...]} with the same length as GT list."
    )
    user_prompt = (
        "Ground-truth errors:\n"
        + json.dumps(gt_errors, ensure_ascii=False)
        + "\n\nPredicted findings:\n"
        + json.dumps(findings, ensure_ascii=False)
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        data = json.loads(raw)
        covered = data.get("covered") if isinstance(data, dict) else None
        if not isinstance(covered, list):
            return None
        out: List[bool] = []
        for i in range(len(gt_errors)):
            v = covered[i] if i < len(covered) else False
            out.append(bool(v))
        return out
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recall/precision on custom physics eval sets.")
    parser.add_argument("--recall-dataset", type=str, required=True)
    parser.add_argument("--precision-dataset", type=str, required=True)
    parser.add_argument("--recall-results", type=str, required=True)
    parser.add_argument("--recall-audit", type=str, required=True)
    parser.add_argument("--precision-results", type=str, required=True)
    parser.add_argument("--precision-audit", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--semantic-match-model", type=str, default="gemini3_pro_preview")
    parser.add_argument("--disable-llm-match", action="store_true")
    args = parser.parse_args()

    recall_ds = _load_json(args.recall_dataset)
    precision_ds = _load_json(args.precision_dataset)
    recall_pred = _load_json(args.recall_results)
    recall_audit = _load_json(args.recall_audit)
    precision_pred = _load_json(args.precision_results)
    precision_audit = _load_json(args.precision_audit)

    if not isinstance(recall_ds, list) or not isinstance(precision_ds, list):
        raise SystemExit("Dataset files must be JSON arrays.")

    recall_pred_idx = _index_by_id(recall_pred if isinstance(recall_pred, list) else [])
    recall_audit_idx = _index_by_id(recall_audit if isinstance(recall_audit, list) else [])
    precision_pred_idx = _index_by_id(precision_pred if isinstance(precision_pred, list) else [])
    precision_audit_idx = _index_by_id(precision_audit if isinstance(precision_audit, list) else [])

    recall_rows: List[Dict[str, Any]] = []
    total_gt_errors = 0
    matched_gt_errors = 0
    llm_match_used = 0

    for row in recall_ds:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "").strip()
        gt_items = row.get("physics_error_examples")
        gt_items = gt_items if isinstance(gt_items, list) else []
        gt_errors = [str(x.get("error") or "").strip() for x in gt_items if isinstance(x, dict) and str(x.get("error") or "").strip()]

        pred_item = recall_pred_idx.get(sid, {})
        audit_item = recall_audit_idx.get(sid, {})
        findings = _collect_pred_findings(pred_item, audit_item)

        covered: List[bool]
        if (not args.disable_llm_match) and gt_errors and findings:
            cov = _llm_match_coverage(args.semantic_match_model, gt_errors, findings)
            if cov is not None:
                covered = cov
                llm_match_used += 1
            else:
                covered = [_keyword_match(e, findings) for e in gt_errors]
        else:
            covered = [_keyword_match(e, findings) for e in gt_errors]

        total_gt_errors += len(gt_errors)
        matched = sum(1 for x in covered if x)
        matched_gt_errors += matched

        recall_rows.append(
            {
                "id": sid,
                "gt_error_count": len(gt_errors),
                "pred_finding_count": len(findings),
                "matched_error_count": matched,
                "sample_error_recall": (matched / len(gt_errors)) if gt_errors else 0.0,
                "pred_has_diagnostic": bool((pred_item or {}).get("diagnostics")),
            }
        )

    total_recall_samples = len(recall_rows)
    recall_sample_triggered = sum(1 for r in recall_rows if r.get("pred_has_diagnostic") is True)

    precision_rows: List[Dict[str, Any]] = []
    fp_samples = 0
    for row in precision_ds:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "").strip()
        pred_item = precision_pred_idx.get(sid, {})
        audit_item = precision_audit_idx.get(sid, {})
        findings = _collect_pred_findings(pred_item, audit_item)

        has_error = bool((pred_item or {}).get("diagnostics")) or bool(findings)
        if has_error:
            fp_samples += 1

        precision_rows.append(
            {
                "id": sid,
                "pred_has_error": has_error,
                "pred_finding_count": len(findings),
            }
        )

    total_precision_samples = len(precision_rows)
    inaccuracy_ratio = (fp_samples / total_precision_samples) if total_precision_samples else 0.0

    output = {
        "summary": {
            "recall_dataset_size": total_recall_samples,
            "precision_dataset_size": total_precision_samples,
            "recall_total_gt_errors": total_gt_errors,
            "recall_matched_gt_errors": matched_gt_errors,
            "recall_error_level": (matched_gt_errors / total_gt_errors) if total_gt_errors else 0.0,
            "recall_sample_trigger_ratio": (recall_sample_triggered / total_recall_samples) if total_recall_samples else 0.0,
            "precision_inaccuracy_ratio": inaccuracy_ratio,
            "precision_proxy": 1.0 - inaccuracy_ratio,
            "llm_semantic_match_used_samples": llm_match_used,
        },
        "recall_details": recall_rows,
        "precision_details": precision_rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
