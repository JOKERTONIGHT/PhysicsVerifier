from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


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


def _collect_pred_findings(pred_item: Dict[str, Any], audit_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    diagnostics = pred_item.get("diagnostics") if isinstance(pred_item, dict) else []
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    for d in diagnostics:
        if not isinstance(d, dict):
            continue
        rule = str(d.get("rule") or "").strip()
        message = str(d.get("message") or "").strip()
        evidence = d.get("evidence")
        quote = ""
        if isinstance(evidence, dict):
            quote = str(evidence.get("quote") or "").strip()
        elif isinstance(evidence, str):
            quote = evidence.strip()

        text = " | ".join([x for x in [rule, message, quote] if x])
        if text:
            findings.append(
                {
                    "source": "diagnostic",
                    "rule": rule,
                    "message": message,
                    "quote": quote,
                    "text": text,
                }
            )

    checks = audit_item.get("experience_code_checks") if isinstance(audit_item, dict) else []
    checks = checks if isinstance(checks, list) else []
    for c in checks:
        if not isinstance(c, dict):
            continue
        result = str(c.get("result") or "").strip().lower()
        if result != "fail":
            continue
        rule = str(c.get("rule") or "").strip()
        message = str(c.get("message") or "").strip()
        evidence = str(c.get("evidence") or "").strip()
        text = " | ".join([x for x in [rule, message, evidence] if x])
        if text:
            findings.append(
                {
                    "source": "experience_code",
                    "rule": rule,
                    "message": message,
                    "quote": "",
                    "text": text,
                }
            )

    out: List[Dict[str, Any]] = []
    seen = set()
    for f in findings:
        key = (str(f.get("source") or ""), str(f.get("rule") or ""), str(f.get("message") or ""), str(f.get("quote") or ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate question-level metrics on mixed positive/negative dataset.")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--results", type=str, required=True)
    parser.add_argument("--audit", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    ds = _load_json(args.dataset)
    pred = _load_json(args.results)
    audit = _load_json(args.audit)

    if not isinstance(ds, list):
        raise SystemExit("Dataset file must be a JSON array.")

    pred_idx = _index_by_id(pred if isinstance(pred, list) else [])
    audit_idx = _index_by_id(audit if isinstance(audit, list) else [])

    tp = 0
    fp = 0
    tn = 0
    fn = 0
    details: List[Dict[str, Any]] = []

    for row in ds:
        if not isinstance(row, dict):
            continue
        sid = str(row.get("id") or "").strip()
        if not sid:
            continue

        expected_has_error = bool(row.get("expected_has_physics_error"))
        pred_item = pred_idx.get(sid, {})
        audit_item = audit_idx.get(sid, {})
        findings = _collect_pred_findings(pred_item, audit_item)
        pred_has_error = bool(findings)

        if expected_has_error and pred_has_error:
            tp += 1
        elif expected_has_error and (not pred_has_error):
            fn += 1
        elif (not expected_has_error) and pred_has_error:
            fp += 1
        else:
            tn += 1

        details.append(
            {
                "id": sid,
                "eval_split": str(row.get("eval_split") or ""),
                "expected_has_physics_error": expected_has_error,
                "pred_has_error": pred_has_error,
                "pred_finding_count": len(findings),
                "decision_source": "diagnostics_or_audit",
            }
        )

    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    output = {
        "summary": {
            "level": "question",
            "dataset_size": len(details),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "recall": recall,
            "precision": precision,
            "f1": f1,
            "precision_proxy": 1.0 - (fp / (fp + tn)) if (fp + tn) else 0.0,
        },
        "details": details,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
