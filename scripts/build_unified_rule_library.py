"""Compatibility builder for the older flat unified catalog format.

For the reusable hierarchical rule framework, use `scripts/manage_rule_library.py
build` or import `rule_framework.builder` directly. This script preserves the
old CLI and output shape used by existing experiments.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rule_framework.builder import build_simple_unified_library  # noqa: E402,F401
from rule_framework.io import load_json, write_json  # noqa: E402


def build_unified_library(
    rules_catalog: Dict[str, Any],
    distilled: Dict[str, Any],
    *,
    rule_source: str = "hybrid",
) -> Dict[str, Any]:
    return build_simple_unified_library(rules_catalog, distilled, rule_source=rule_source)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build legacy flat unified rule library from top-down catalog and distilled experience rules.")
    parser.add_argument("--rules-catalog", type=str, default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--experience-distilled", type=str, required=True)
    parser.add_argument("--output", type=str, default="catalogs/unified_rule_library.json")
    parser.add_argument(
        "--rule-source",
        type=str,
        default="experience-only",
        choices=["experience-only", "hybrid", "knowledge-only"],
        help="Which rule set to emit into final verifier-compatible catalog.",
    )
    args = parser.parse_args()

    catalog = load_json(Path(args.rules_catalog))
    distilled = load_json(Path(args.experience_distilled))
    unified = build_unified_library(catalog, distilled, rule_source=args.rule_source)

    out_path = Path(args.output)
    write_json(out_path, unified)
    print(f"Done. Unified rule library saved to {out_path}")
    meta = unified.get("metadata", {}) if isinstance(unified, dict) else {}
    print(
        "Summary:",
        json.dumps(
            {
                "rule_source": meta.get("rule_source"),
                "total_domains": meta.get("total_domains"),
                "total_topics": meta.get("total_topics"),
                "total_rules": meta.get("total_rules"),
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
