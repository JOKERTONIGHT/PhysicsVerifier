from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, List, Tuple


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if t]


def _is_generic_error_text(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s:
        return True
    generic_markers = ["should", "must", "should be", "not satisfy", "violate", "incorrect"]
    marker_hit = any(m in s for m in generic_markers)
    tok = _tokenize(s)
    physics_markers = ["force", "energy", "momentum", "voltage", "current", "resistance", "field", "acceleration", "velocity", "work", "power", "entropy", "temperature", "pressure"]
    has_physics_term = any(m in s for m in physics_markers)
    return bool(marker_hit and not has_physics_term and len(tok) < 12)


def _audit_recall_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample_count = 0
    error_count_list: List[int] = []

    total_errors = 0
    total_quote_non_empty = 0
    total_span_valid = 0
    total_para_valid = 0
    total_locatable_valid = 0
    total_generic = 0
    total_short_error = 0

    samples_zero_errors: List[str] = []
    samples_low_locatable: List[str] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_count += 1
        sid = str(row.get("id") or "")

        gt_items = row.get("physics_error_gt")
        gt_items = gt_items if isinstance(gt_items, list) else []

        error_count = len(gt_items)
        error_count_list.append(error_count)
        total_errors += error_count

        if error_count == 0 and sid:
            samples_zero_errors.append(sid)

        loc_cnt = 0
        for g in gt_items:
            if not isinstance(g, dict):
                continue
            err = str(g.get("error_text") or g.get("error") or "")
            quote = str(g.get("answer_quote") or g.get("quote") or "")
            span_valid = bool(g.get("span_valid"))
            para_valid = bool(g.get("paragraph_valid"))
            loc_valid = bool(g.get("locatable_valid")) or bool(span_valid or para_valid)

            if quote.strip():
                total_quote_non_empty += 1
            if span_valid:
                total_span_valid += 1
            if para_valid:
                total_para_valid += 1
            if loc_valid:
                total_locatable_valid += 1
                loc_cnt += 1
            if _is_generic_error_text(err):
                total_generic += 1
            if len(_tokenize(err)) <= 4:
                total_short_error += 1

        if error_count > 0 and loc_cnt == 0 and sid:
            samples_low_locatable.append(sid)

    avg_errors = mean(error_count_list) if error_count_list else 0.0
    med_errors = median(error_count_list) if error_count_list else 0.0

    quote_ratio = (total_quote_non_empty / total_errors) if total_errors else 0.0
    span_ratio = (total_span_valid / total_errors) if total_errors else 0.0
    para_ratio = (total_para_valid / total_errors) if total_errors else 0.0
    loc_ratio = (total_locatable_valid / total_errors) if total_errors else 0.0
    generic_ratio = (total_generic / total_errors) if total_errors else 0.0
    short_ratio = (total_short_error / total_errors) if total_errors else 0.0

    return {
        "sample_count": sample_count,
        "total_errors": total_errors,
        "avg_errors_per_sample": avg_errors,
        "median_errors_per_sample": med_errors,
        "quote_non_empty_ratio": quote_ratio,
        "span_valid_ratio": span_ratio,
        "paragraph_valid_ratio": para_ratio,
        "locatable_valid_ratio": loc_ratio,
        "generic_error_ratio": generic_ratio,
        "short_error_ratio": short_ratio,
        "samples_with_zero_errors": samples_zero_errors,
        "samples_with_no_locatable_errors": samples_low_locatable,
    }


def _quality_gate(summary: Dict[str, Any], min_loc_ratio: float, min_avg_errors: float, max_generic_ratio: float) -> Tuple[bool, List[str]]:
    issues: List[str] = []

    avg_errors = float(summary.get("avg_errors_per_sample") or 0.0)
    loc_ratio = float(summary.get("locatable_valid_ratio") or 0.0)
    generic_ratio = float(summary.get("generic_error_ratio") or 0.0)
    quote_ratio = float(summary.get("quote_non_empty_ratio") or 0.0)
    zero_count = len(summary.get("samples_with_zero_errors") or [])

    if avg_errors < min_avg_errors:
        issues.append(f"avg_errors_per_sample too low: {avg_errors:.3f} < {min_avg_errors:.3f}")
    if loc_ratio < min_loc_ratio:
        issues.append(f"locatable_valid_ratio too low: {loc_ratio:.3f} < {min_loc_ratio:.3f}")
    if generic_ratio > max_generic_ratio:
        issues.append(f"generic_error_ratio too high: {generic_ratio:.3f} > {max_generic_ratio:.3f}")
    if quote_ratio < 0.70:
        issues.append(f"quote_non_empty_ratio too low: {quote_ratio:.3f} < 0.700")
    if zero_count > 0:
        issues.append(f"samples_with_zero_errors found: {zero_count}")

    return (len(issues) == 0), issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit generated evaluation set quality for completeness/location usability.")
    parser.add_argument("--recall-dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--min-locatable-ratio", type=float, default=0.70)
    parser.add_argument("--min-avg-errors", type=float, default=2.0)
    parser.add_argument("--max-generic-ratio", type=float, default=0.25)
    args = parser.parse_args()

    recall_data = _load_json(args.recall_dataset)
    if not isinstance(recall_data, list):
        raise SystemExit("Recall dataset must be a JSON array")

    summary = _audit_recall_dataset(recall_data)
    ok, issues = _quality_gate(
        summary,
        min_loc_ratio=float(args.min_locatable_ratio),
        min_avg_errors=float(args.min_avg_errors),
        max_generic_ratio=float(args.max_generic_ratio),
    )

    out = {
        "quality_gate_passed": ok,
        "quality_gate_issues": issues,
        "summary": summary,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
