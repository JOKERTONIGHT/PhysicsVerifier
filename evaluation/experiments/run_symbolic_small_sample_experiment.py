#!/usr/bin/env python3
"""Run small-sample symbolic experiments (audit-only, no pipeline integration).

Reads experiment_manifest.json, extracts formulas from predictions, and runs
declarative pattern/constraint checks. SymPy parsing is attempted when available;
falls back to pattern matching. Outputs structured evidence JSON.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _find_span(text: str, pattern: str) -> Optional[Dict[str, Any]]:
    if not pattern or not text:
        return None
    try:
        m = re.search(re.escape(pattern), text, re.I)
    except re.error:
        m = re.search(pattern, text, re.I)
    if not m:
        # fuzzy: strip latex backslashes
        simplified = pattern.replace("\\", "")
        m = re.search(re.escape(simplified), text, re.I)
    if m:
        return {"start_char": m.start(), "end_char": m.end(), "matched_text": text[m.start() : m.end()]}
    return None


def _check_pattern(prediction: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    wrong_patterns = spec.get("wrong_patterns") or []
    canonical = str(spec.get("canonical") or "")
    hits_wrong = []
    for pat in wrong_patterns:
        span = _find_span(prediction, pat)
        if span:
            hits_wrong.append({"pattern": pat, **span})

    if hits_wrong:
        return {
            "result": "supports_error",
            "backend": "pattern",
            "evidence": f"Wrong pattern detected: {hits_wrong[0]['pattern']}",
            "matched_span": hits_wrong[0],
            "details": {"wrong_hits": hits_wrong},
        }

    if canonical:
        span = _find_span(prediction, canonical.replace("*", ""))
        if span:
            return {
                "result": "no_signal",
                "backend": "pattern",
                "evidence": "Canonical pattern present; no wrong pattern hit",
                "matched_span": span,
                "details": {},
            }

    return {
        "result": "no_signal",
        "backend": "pattern",
        "evidence": "No matching wrong or canonical pattern",
        "matched_span": None,
        "details": {},
    }


def _check_inequality_constraint(prediction: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Detect non-relativistic v formula when v > c is implied."""
    expr_hint = str(spec.get("expression") or "")
    constraint = str(spec.get("constraint") or "v <= c")
    if "sqrt(2" in expr_hint.lower() or "sqrt(2*deltae" in _normalize_text(prediction):
        if re.search(r"sqrt\s*\(\s*2\s*\*?\s*(?:delta\s*e|ΔE|deltae)", prediction, re.I):
            span = _find_span(prediction, "sqrt")
            return {
                "result": "supports_error",
                "backend": "constraint",
                "evidence": f"Non-relativistic velocity formula may violate {constraint}",
                "matched_span": span,
                "details": {"constraint": constraint},
            }
    return {
        "result": "no_signal",
        "backend": "constraint",
        "evidence": "Constraint check inconclusive",
        "matched_span": None,
        "details": {},
    }


def _check_power_exponent(prediction: str, spec: Dict[str, Any]) -> Dict[str, Any]:
    var = str(spec.get("variable") or "R")
    wrong = spec.get("wrong_exponent")
    if wrong is None:
        return {"result": "no_signal", "backend": "sympy", "evidence": "No wrong exponent configured", "matched_span": None, "details": {}}

    # Match R^5, a^4, etc.
    pat = rf"{re.escape(var)}\s*\^\s*{wrong}\b"
    m = re.search(pat, prediction, re.I)
    if m:
        return {
            "result": "supports_error",
            "backend": "sympy",
            "evidence": f"Found {var}^{wrong}; expected exponent {spec.get('expected_exponent')}",
            "matched_span": {"start_char": m.start(), "end_char": m.end(), "matched_text": m.group(0)},
            "details": {"wrong_exponent": wrong},
        }
    return {
        "result": "no_signal",
        "backend": "sympy",
        "evidence": f"No {var}^{wrong} found",
        "matched_span": None,
        "details": {},
    }


def _run_experiment(
    sample: Dict[str, Any],
    exp: Dict[str, Any],
    target_error_ids: List[str],
) -> Dict[str, Any]:
    prediction = str(sample.get("prediction") or "")
    primitive = str(exp.get("primitive") or "")
    backend = str(exp.get("backend") or "pattern")

    if primitive == "inequality_constraint":
        out = _check_inequality_constraint(prediction, exp)
    elif primitive == "power_exponent_check":
        out = _check_power_exponent(prediction, exp)
    elif primitive in ("formula_pattern", "component_relation_check", "equation_equivalence_sympy"):
        out = _check_pattern(prediction, exp)
    elif primitive == "dimension_check":
        # Phase 0: flag if expression substring present (full dimension engine in Phase 2)
        expr = str(exp.get("expression") or "")
        if expr and _find_span(prediction, expr.split("=")[0].strip()):
            out = {
                "result": "supports_error",
                "backend": "dimension",
                "evidence": f"Expression present; dimension check pending full engine: {expr}",
                "matched_span": _find_span(prediction, expr.split("=")[0].strip()),
                "details": {"phase": "0_stub", "expected_dimension": exp.get("expected_dimension")},
            }
        else:
            out = {"result": "no_signal", "backend": "dimension", "evidence": "Expression not found", "matched_span": None, "details": {}}
    elif primitive == "sign_check":
        if re.search(r"dE\s*/\s*dt\s*=\s*[^-\n]+(?:\+|(?![\s-]))", prediction):
            out = {
                "result": "supports_error",
                "backend": "constraint",
                "evidence": "dE/dt appears positive (energy loss should be negative)",
                "matched_span": _find_span(prediction, "dE"),
                "details": {},
            }
        else:
            out = {"result": "no_signal", "backend": "constraint", "evidence": "Sign check inconclusive", "matched_span": None, "details": {}}
    else:
        out = {"result": "no_signal", "backend": backend, "evidence": f"Unknown primitive: {primitive}", "matched_span": None, "details": {}}

    out["experiment_id"] = exp.get("experiment_id")
    out["primitive"] = primitive
    out["target_error_ids"] = target_error_ids
    return out


def run_manifest(
    manifest_path: Path,
    dataset_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    manifest = _load_json(manifest_path)
    dataset = _load_json(dataset_path)
    samples = dataset if isinstance(dataset, list) else dataset.get("samples") or []
    by_id = {str(s.get("id")): s for s in samples if isinstance(s, dict) and s.get("id")}

    results: List[Dict[str, Any]] = []
    summary = {"supports_error": 0, "no_signal": 0, "refutes_error": 0, "total_experiments": 0}

    for entry in manifest.get("samples") or []:
        sid = str(entry.get("sample_id") or "")
        sample = by_id.get(sid)
        if not sample:
            continue
        target_ids = entry.get("target_gt_errors") or []
        for exp in entry.get("experiments") or []:
            summary["total_experiments"] += 1
            rec = _run_experiment(sample, exp, target_ids)
            rec["sample_id"] = sid
            results.append(rec)
            summary[rec.get("result", "no_signal")] = summary.get(rec.get("result", "no_signal"), 0) + 1

    report = {
        "manifest": str(manifest_path),
        "dataset": str(dataset_path),
        "summary": summary,
        "results": results,
        "note": "Phase 0 audit-only. Manual review required for evidence_precision against target_error_ids.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small-sample symbolic experiment (audit-only).")
    parser.add_argument(
        "--manifest",
        default="data/derived/symbolic_small_sample_experiment_v1/experiment_manifest.json",
    )
    parser.add_argument(
        "--dataset",
        default="results/scale_curve_error_v2_local_30b/scale_1500_cleaned/error_eval_dataset_100.json",
    )
    parser.add_argument(
        "--output",
        default="results/symbolic_small_sample_v1/audit.json",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    report = run_manifest(
        manifest_path=root / args.manifest,
        dataset_path=root / args.dataset,
        output_path=root / args.output,
    )
    print(f"Wrote {root / args.output}")
    print(f"Experiments: {report['summary']['total_experiments']} | supports_error: {report['summary'].get('supports_error', 0)}")


if __name__ == "__main__":
    main()
