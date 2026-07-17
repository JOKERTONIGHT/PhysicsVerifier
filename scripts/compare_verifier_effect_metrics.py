from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: str) -> Dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SystemExit(f"{path} must be a JSON object.")
    return data


def _summary(data: Dict[str, Any]) -> Dict[str, Any]:
    summary = data.get("summary")
    if isinstance(summary, dict):
        return summary
    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        return metrics
    return data


def _num(data: Dict[str, Any], key: str) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _metric_delta(before: Dict[str, Any], after: Dict[str, Any], key: str) -> Dict[str, Any]:
    b = _num(before, key)
    a = _num(after, key)
    if b is None or a is None:
        return {"before": b, "after": a, "delta": None}
    return {"before": b, "after": a, "delta": a - b}


def _int_delta(before: Dict[str, Any], after: Dict[str, Any], key: str) -> Dict[str, Any]:
    b = _num(before, key)
    a = _num(after, key)
    if b is None or a is None:
        return {"before": b, "after": a, "delta": None}
    return {"before": int(b), "after": int(a), "delta": int(a - b)}


def _compare_one(before_path: str, after_path: str, *, level: str) -> Dict[str, Any]:
    before = _summary(_load_json(before_path))
    after = _summary(_load_json(after_path))
    metric_keys = ["precision", "recall", "f1", "accuracy", "precision_proxy", "recall_location_only"]
    count_keys = [
        "tp",
        "fp",
        "tn",
        "fn",
        "total_gt_errors",
        "matched_gt_errors",
        "total_pred_locatable_findings",
        "matched_pred_locatable",
        "location_unmatched_pred_findings",
    ]
    return {
        "level": level,
        "before": before_path,
        "after": after_path,
        "metrics": {key: _metric_delta(before, after, key) for key in metric_keys if key in before or key in after},
        "counts": {key: _int_delta(before, after, key) for key in count_keys if key in before or key in after},
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Verifier Effect Comparison")
    lines.append("")
    for section in report.get("comparisons", []):
        level = section.get("level")
        lines.append(f"## {level}")
        lines.append("")
        lines.append("| type | name | before | after | delta |")
        lines.append("|---|---|---:|---:|---:|")
        for group_name in ("metrics", "counts"):
            group = section.get(group_name) if isinstance(section.get(group_name), dict) else {}
            for key, payload in group.items():
                lines.append(
                    f"| {group_name} | {key} | {_fmt(payload.get('before'))} | "
                    f"{_fmt(payload.get('after'))} | {_fmt(payload.get('delta'))} |"
                )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare before/after verifier effect metrics.")
    parser.add_argument("--before-error", default="")
    parser.add_argument("--after-error", default="")
    parser.add_argument("--before-question", default="")
    parser.add_argument("--after-question", default="")
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    parser.add_argument("--markdown-output", default="", help="Optional Markdown summary path.")
    args = parser.parse_args()

    comparisons: List[Dict[str, Any]] = []
    if args.before_error and args.after_error:
        comparisons.append(_compare_one(args.before_error, args.after_error, level="error"))
    if args.before_question and args.after_question:
        comparisons.append(_compare_one(args.before_question, args.after_question, level="question"))
    if not comparisons:
        raise SystemExit("Provide at least one before/after pair.")

    report = {"comparisons": comparisons}
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.markdown_output:
        out_md = Path(args.markdown_output)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_markdown(report), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
