#!/usr/bin/env python3
"""Analyze semantic checker results and identify symbolizable error candidates.

Reads pure-semantic and rules-baseline evaluation artifacts, classifies GT errors
by theme and failure cause, and emits a JSON manifest for small-sample symbolic
experiments. Does not run symbolic verification.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# Keywords suggesting formula/algebra/dimension errors (symbolizable).
_SYMBOLIZABLE_PATTERNS: List[Tuple[str, str]] = [
    (r"dimensionally|量纲", "dimension_check"),
    (r"incorrect (formula|expression|derivative|integral|resistance|velocity|energy|exponent|power|sign|factor|coefficient)", "equation_equivalence_sympy"),
    (r"wrong (sign|power|exponent|factor|formula|expression|derivative|integral)", "equation_equivalence_sympy"),
    (r"should be .* instead of", "equation_equivalence_sympy"),
    (r"sin|cos|tan|component", "component_relation_check"),
    (r"greater than the speed of light|v\s*[<>]\s*c|v\s*>\s*c", "inequality_constraint"),
    (r"\^4|\^5|power of|exponent", "power_exponent_check"),
    (r"positive \(should be negative\)|negative \(should be positive\)|sign error", "sign_check"),
    (r"R\^5|a\^4|a\^2|\^", "power_exponent_check"),
    (r"sqrt\(|factor of", "equation_equivalence_sympy"),
]

# Keywords suggesting non-symbolizable (modeling/concept) errors.
_NON_SYMBOLIZABLE_PATTERNS: List[str] = [
    r"incorrectly assumes",
    r"misinterpretation",
    r"does not use the given",
    r"non sequitur",
    r"direction assignment",
    r"leaving to the (inside|outside)",
    r"modeling",
    r"conceptual",
    r"closed surface is zero",
    r"lumped-circuit",
    r"distributed system",
    r"numerical value.*assuming",
    r"wavelength.*not specify",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _classify_error_text(text: str) -> Tuple[str, List[str]]:
    """Return (primary_category, suggested_primitives)."""
    lower = text.lower()
    for pat in _NON_SYMBOLIZABLE_PATTERNS:
        if re.search(pat, lower, re.I):
            return "non_symbolizable", []

    primitives: List[str] = []
    for pat, primitive in _SYMBOLIZABLE_PATTERNS:
        if re.search(pat, lower, re.I):
            if primitive not in primitives:
                primitives.append(primitive)

    if primitives:
        return "symbolizable", primitives
    return "uncertain", ["equation_equivalence_sympy"]


def _infer_theme(error_text: str, sample_id: str) -> str:
    t = error_text.lower()
    if any(k in t for k in ("orbit", "gravit", "slingshot", "binary")):
        return "orbital_gravity"
    if any(k in t for k in ("derivative", "integral", "algebra", "dimensionally", "factor")):
        return "formula_algebra"
    if any(k in t for k in ("resistance", "heat", "temperature", "thermal")):
        return "thermo_fluid"
    if any(k in t for k in ("relativ", "cherenkov", "speed of light", "lorentz")):
        return "relativity_optics"
    if any(k in t for k in ("magnetic", "flux", "emf", "eddy", "induct")):
        return "electromagnetism"
    if any(k in t for k in ("wave vector", "evanescent", "sin θ", "cos θ", "wavelength")):
        return "waves_optics"
    return "other"


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", text.lower()))


def _match_semantic_status(
    gt_error: Dict[str, Any],
    diagnostics: List[Dict[str, Any]],
) -> str:
    """Classify whether pure semantic checker matched this GT."""
    gt_tokens = _tokenize(str(gt_error.get("error_text") or gt_error.get("quote") or ""))
    if not gt_tokens or not diagnostics:
        return "semantic_gap"

    best_overlap = 0
    for d in diagnostics:
        if not isinstance(d, dict):
            continue
        ev = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
        blob = " ".join(
            str(x)
            for x in (
                d.get("message"),
                ev.get("quote"),
            )
            if x
        )
        pred_tokens = _tokenize(blob)
        overlap = len(gt_tokens & pred_tokens)
        best_overlap = max(best_overlap, overlap)

    if best_overlap >= 8:
        return "semantic_hit"
    if best_overlap >= 3:
        return "semantic_near_miss"
    return "semantic_gap"


def analyze(
    dataset_path: Path,
    checker_results_path: Optional[Path],
    matching_sensitivity_path: Optional[Path],
    failure_by_rule_path: Optional[Path],
) -> Dict[str, Any]:
    dataset = _load_json(dataset_path)
    samples = dataset if isinstance(dataset, list) else dataset.get("samples") or []

    checker_by_id: Dict[str, Dict[str, Any]] = {}
    if checker_results_path and checker_results_path.exists():
        checker_data = _load_json(checker_results_path)
        for item in checker_data if isinstance(checker_data, list) else checker_data.get("results") or []:
            if isinstance(item, dict) and item.get("id"):
                checker_by_id[str(item["id"])] = item

    sensitivity: Dict[str, Any] = {}
    if matching_sensitivity_path and matching_sensitivity_path.exists():
        sens = _load_json(matching_sensitivity_path)
        for exp in sens.get("experiments") or []:
            if isinstance(exp, dict):
                sensitivity[str(exp.get("label") or "")] = exp

    failure_themes: Dict[str, Any] = {}
    if failure_by_rule_path and failure_by_rule_path.exists():
        fbr = _load_json(failure_by_rule_path)
        for theme_block in fbr.get("top_missed_gt_themes") or []:
            if isinstance(theme_block, dict):
                failure_themes[str(theme_block.get("theme") or "")] = theme_block

    all_errors: List[Dict[str, Any]] = []
    theme_counts: Counter = Counter()
    category_counts: Counter = Counter()
    semantic_status_counts: Counter = Counter()

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sid = str(sample.get("id") or "")
        gt_list = sample.get("physics_error_gt") or []
        checker_item = checker_by_id.get(sid) or {}
        diagnostics = checker_item.get("diagnostics") or []

        for gt in gt_list:
            if not isinstance(gt, dict):
                continue
            error_text = str(gt.get("error_text") or gt.get("quote") or "")
            category, primitives = _classify_error_text(error_text)
            theme = _infer_theme(error_text, sid)
            semantic_status = _match_semantic_status(gt, diagnostics)

            theme_counts[theme] += 1
            category_counts[category] += 1
            semantic_status_counts[semantic_status] += 1

            rec = {
                "sample_id": sid,
                "error_id": str(gt.get("error_id") or ""),
                "error_text": error_text[:200],
                "theme": theme,
                "category": category,
                "semantic_status": semantic_status,
                "suggested_primitives": primitives,
                "paragraph_index": gt.get("paragraph_index"),
            }
            all_errors.append(rec)

    # Priority: symbolizable + semantic_gap, then symbolizable + near_miss in high-gap themes
    high_gap_themes = {"formula_algebra", "orbital_gravity", "electromagnetism", "waves_optics"}

    def priority_key(e: Dict[str, Any]) -> Tuple[int, int, str]:
        cat_score = {"symbolizable": 0, "uncertain": 1, "non_symbolizable": 2}[e["category"]]
        sem_score = {"semantic_gap": 0, "semantic_near_miss": 1, "semantic_hit": 2}[e["semantic_status"]]
        theme_bonus = 0 if e["theme"] in high_gap_themes else 1
        return (cat_score, sem_score + theme_bonus, e["error_id"])

    ranked = sorted(all_errors, key=priority_key)

    # Curated first-batch sample IDs from plan
    curated_sample_ids = {
        "cl_209_132531",
        "cl_104_24899",
        "cl_188_110801",
        "cl_110_31637",
        "cl_172_95214",
        "cl_132_53961",
    }

    curated_errors = [e for e in all_errors if e["sample_id"] in curated_sample_ids]
    top_candidates = [e for e in ranked if e["category"] == "symbolizable"][:40]

    return {
        "meta": {
            "dataset": str(dataset_path),
            "checker_results": str(checker_results_path) if checker_results_path else None,
            "total_samples": len(samples),
            "total_gt_errors": len(all_errors),
        },
        "metrics_summary": {
            "matching_sensitivity": sensitivity,
            "failure_themes": failure_themes,
        },
        "aggregate": {
            "by_category": dict(category_counts),
            "by_theme": dict(theme_counts),
            "by_semantic_status": dict(semantic_status_counts),
            "symbolizable_and_gap": sum(
                1 for e in all_errors if e["category"] == "symbolizable" and e["semantic_status"] == "semantic_gap"
            ),
            "symbolizable_and_near_miss": sum(
                1
                for e in all_errors
                if e["category"] == "symbolizable" and e["semantic_status"] == "semantic_near_miss"
            ),
        },
        "curated_samples": {
            "sample_ids": sorted(curated_sample_ids),
            "errors": curated_errors,
            "symbolizable_count": sum(1 for e in curated_errors if e["category"] == "symbolizable"),
            "total_count": len(curated_errors),
        },
        "top_symbolizable_candidates": top_candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze symbolizable error candidates from semantic eval.")
    parser.add_argument(
        "--dataset",
        default="results/scale_curve_error_v2_local_30b/scale_1500_cleaned/error_eval_dataset_100.json",
    )
    parser.add_argument(
        "--checker-results",
        default="results/semantic_pure_check_cleaned_1500/local_30b/checker_results.json",
    )
    parser.add_argument(
        "--matching-sensitivity",
        default="results/recall_attribution_v1/final_report/matching_sensitivity_report.json",
    )
    parser.add_argument(
        "--failure-by-rule",
        default="results/scale_curve_error_v2_local_30b/scale_1500_cleaned/failure_analysis_by_rule.json",
    )
    parser.add_argument(
        "--output",
        default="data/derived/symbolic_small_sample_experiment_v1/candidate_errors.json",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    out_path = root / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = analyze(
        dataset_path=root / args.dataset,
        checker_results_path=root / args.checker_results if args.checker_results else None,
        matching_sensitivity_path=root / args.matching_sensitivity if args.matching_sensitivity else None,
        failure_by_rule_path=root / args.failure_by_rule if args.failure_by_rule else None,
    )
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    agg = report["aggregate"]
    print(
        f"GT errors: {report['meta']['total_gt_errors']} | "
        f"symbolizable: {agg['by_category'].get('symbolizable', 0)} | "
        f"symbolizable+gap: {agg['symbolizable_and_gap']}"
    )


if __name__ == "__main__":
    main()
