from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_id(value: Any) -> str:
    return str(value).strip()


def _compute_confusion(records: List[Dict[str, Any]]) -> Dict[str, int]:
    tp = fp = fn = tn = 0
    for r in records:
        pred_pos = bool(r.get("pred_has_diagnostic"))
        gt_pos = bool(r.get("gt_is_incorrect"))
        if pred_pos and gt_pos:
            tp += 1
        elif pred_pos and not gt_pos:
            fp += 1
        elif (not pred_pos) and gt_pos:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _compute_prf1(conf: Dict[str, int]) -> Dict[str, float]:
    tp = conf["tp"]
    fp = conf["fp"]
    fn = conf["fn"]
    tn = conf["tn"]

    total = tp + fp + fn + tn
    precision = (tp / (tp + fp)) if (tp + fp) else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = ((tp + tn) / total) if total else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def _build_audit_index(audit_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for item in audit_items:
        if not isinstance(item, dict):
            continue
        sid = _safe_id(item.get("id"))
        if not sid:
            continue
        out[sid] = item
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute strict rubric-based metrics for verifier outputs.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to run_top_down main output JSON.")
    parser.add_argument("--audit", type=str, required=True, help="Path to run_top_down symbolic audit JSON.")
    parser.add_argument("--rubric-meta", type=str, required=True, help="Path to rubric meta JSON from prepare_rubric_eval_subset.py.")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--checkpoint-size", type=int, default=0)
    args = parser.parse_args()

    pred_data = _load_json(Path(args.predictions))
    audit_data = _load_json(Path(args.audit))
    meta_data = _load_json(Path(args.rubric_meta))

    if not isinstance(pred_data, list):
        raise SystemExit("--predictions must be a JSON array.")
    if not isinstance(audit_data, list):
        raise SystemExit("--audit must be a JSON array.")
    if not isinstance(meta_data, dict) or not isinstance(meta_data.get("samples"), list):
        raise SystemExit("--rubric-meta must be a JSON object with field 'samples'.")

    pred_index: Dict[str, Dict[str, Any]] = {}
    for item in pred_data:
        if not isinstance(item, dict):
            continue
        sid = _safe_id(item.get("id"))
        if sid:
            pred_index[sid] = item

    audit_index = _build_audit_index(audit_data)
    meta_samples = meta_data.get("samples") or []

    evaluable_rows: List[Dict[str, Any]] = []

    check_total = 0
    check_fail = 0
    check_inconclusive = 0
    check_pass = 0
    sample_all_inconclusive = 0
    sample_with_checks = 0
    missing_binding_suppressed = 0
    total_suppressed = 0

    for m in meta_samples:
        if not isinstance(m, dict):
            continue
        sid = _safe_id(m.get("id"))
        if not sid:
            continue

        pred_item = pred_index.get(sid, {})
        diagnostics = pred_item.get("diagnostics") if isinstance(pred_item, dict) else []
        pred_has_diagnostic = bool(diagnostics)

        audit_item = audit_index.get(sid, {})
        checks = audit_item.get("experience_code_checks") if isinstance(audit_item, dict) else []
        checks = checks if isinstance(checks, list) else []

        if checks:
            sample_with_checks += 1
            local_has_fail = False
            local_all_inconclusive = True
            for c in checks:
                if not isinstance(c, dict):
                    continue
                res = str(c.get("result") or "").strip().lower()
                if res not in {"pass", "fail", "inconclusive"}:
                    continue
                check_total += 1
                if res == "pass":
                    check_pass += 1
                    local_all_inconclusive = False
                elif res == "fail":
                    check_fail += 1
                    local_has_fail = True
                    local_all_inconclusive = False
                else:
                    check_inconclusive += 1
            if local_all_inconclusive and not local_has_fail:
                sample_all_inconclusive += 1

        suppressed = audit_item.get("suppressed_diagnostics") if isinstance(audit_item, dict) else []
        suppressed = suppressed if isinstance(suppressed, list) else []
        total_suppressed += len(suppressed)
        for s in suppressed:
            if isinstance(s, dict) and str(s.get("reason") or "") == "missing_experience_code_binding":
                missing_binding_suppressed += 1

        is_evaluable = m.get("is_evaluable") is True
        gt_is_incorrect = m.get("gt_is_incorrect")
        if is_evaluable and isinstance(gt_is_incorrect, bool):
            evaluable_rows.append(
                {
                    "id": sid,
                    "pred_has_diagnostic": pred_has_diagnostic,
                    "gt_is_incorrect": gt_is_incorrect,
                    "strict_exact_match": m.get("strict_exact_match"),
                }
            )

    conf = _compute_confusion(evaluable_rows)
    metric = _compute_prf1(conf)

    total_evaluable = len(evaluable_rows)
    gt_incorrect_count = len([r for r in evaluable_rows if r.get("gt_is_incorrect") is True])

    output = {
        "summary": {
            "checkpoint_size": args.checkpoint_size,
            "predictions_file": args.predictions,
            "audit_file": args.audit,
            "rubric_meta_file": args.rubric_meta,
            "total_meta_samples": len(meta_samples),
            "total_evaluable_samples": total_evaluable,
            "gt_incorrect_count": gt_incorrect_count,
        },
        "assumptions": {
            "strict_gt_definition": "ground_truth_normalized 与 prediction_normalized 全等时视为正确，否则视为错误",
            "pred_positive_definition": "run_top_down 输出中 diagnostics 非空",
        },
        "confusion": conf,
        "metrics": metric,
        "code_check_stats": {
            "experience_code_total": check_total,
            "experience_code_fail": check_fail,
            "experience_code_pass": check_pass,
            "experience_code_inconclusive": check_inconclusive,
            "experience_code_inconclusive_ratio": (check_inconclusive / check_total) if check_total else 0.0,
            "samples_with_any_experience_check": sample_with_checks,
            "sample_all_inconclusive_ratio": (sample_all_inconclusive / sample_with_checks) if sample_with_checks else 0.0,
            "suppressed_total": total_suppressed,
            "suppressed_missing_binding": missing_binding_suppressed,
            "suppressed_missing_binding_ratio": (missing_binding_suppressed / total_suppressed) if total_suppressed else 0.0,
        },
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
