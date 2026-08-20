"""Audit the symbolic_hint coverage of a unified rule catalog.

Reports how many rules carry a symbolic_hint, whether the hint is buildable
into an executable spec under the current ``_build_experience_symbolic_spec_from_hint``
contract, and the per-primitive distribution.

Usage:
    .venv/bin/python scripts/audit_symbolic_hints.py \
        --catalog catalogs/legacy/unified_rule_library_v2_llm_enhanced_20260504.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _walk_rules(node: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(node, dict):
        if "rules" in node and isinstance(node["rules"], list) and "topics" not in node:
            for r in node["rules"]:
                if isinstance(r, dict):
                    yield r
        for v in node.values():
            yield from _walk_rules(v)
    elif isinstance(node, list):
        for x in node:
            yield from _walk_rules(x)


def _hint_buildable(primitive: str, canonical: str, syms: List[str]) -> bool:
    if primitive in {"equation_equivalence", "inequality_consistency"}:
        return bool(canonical) and len(syms) >= 2
    if primitive == "formula_pattern":
        return len(syms) >= 2
    if primitive == "power_law":
        return len(syms) >= 2
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    rules = list(_walk_rules(catalog))

    primitive_counts: Counter = Counter()
    buildable_counts: Counter = Counter()
    syms_n_hist: Counter = Counter()
    no_canonical: List[str] = []
    short_syms: List[str] = []
    not_buildable: List[Dict[str, Any]] = []

    for rule in rules:
        rid = str(rule.get("rule_id") or rule.get("id") or "")
        sh = rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {}
        primitive = str(sh.get("primitive") or "none").strip().lower()
        canonical = str(sh.get("canonical") or "").strip()
        syms = [str(s) for s in (sh.get("required_symbols") or []) if str(s).strip()]

        primitive_counts[primitive] += 1
        syms_n_hist[len(syms)] += 1
        if not canonical and primitive in {"equation_equivalence", "inequality_consistency"}:
            no_canonical.append(rid)
        if len(syms) < 2 and primitive in {"equation_equivalence", "inequality_consistency", "formula_pattern", "power_law"}:
            short_syms.append(rid)

        if _hint_buildable(primitive, canonical, syms):
            buildable_counts["buildable"] += 1
        else:
            buildable_counts["not_buildable"] += 1
            not_buildable.append({"rule_id": rid, "primitive": primitive, "syms": syms, "canonical_present": bool(canonical)})

    summary = {
        "total_rules": len(rules),
        "primitive_distribution": dict(primitive_counts),
        "required_symbol_count_distribution": dict(sorted(syms_n_hist.items())),
        "buildable_distribution": dict(buildable_counts),
        "rules_missing_canonical_for_eq_primitive": len(no_canonical),
        "rules_with_short_required_symbols": len(short_syms),
        "examples_not_buildable": not_buildable[:20],
    }

    output = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Wrote audit summary to {args.output}")
    print(output)


if __name__ == "__main__":
    main()
