"""Compatibility CLI for building unified_rules v2 catalogs.

The reusable implementation lives in `rule_framework.builder`; this script is
kept so existing runbooks and tests can keep importing/running
`scripts/merge_rules.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rule_framework.builder import (  # noqa: E402,F401
    CLUSTER_BUCKET_THRESHOLD,
    CLUSTER_TOPIC_THRESHOLD,
    build_unified_catalog,
    build_unified_catalog_from_data,
)
from rule_framework.io import write_json  # noqa: E402


def _print_summary(catalog: Dict[str, Any], output_path: Path) -> None:
    meta = catalog["metadata"]
    print(f"Done. Unified v2 catalog written to: {output_path}")
    print(f"  Domains:             {meta['total_domains']}")
    print(f"  Topics:              {meta['total_topics']}")
    print(f"  Topics with rules:   {meta['topics_with_rules']}")
    print(f"  Executable rules:    {meta['total_executable_rules']}")
    print(f"  Tagged refs mapped:  {meta['mapped_tagged_reference_rules']}")
    print(f"  Clustered topics:    {meta['clustered_topics']}")
    print(f"  Total clusters:      {meta['total_clusters']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified_rules v2 from knowledge skeleton and distilled rules.")
    parser.add_argument("--knowledge", type=str, default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--experience-tagged", type=str, default="catalogs/rules_300_tagged.json")
    parser.add_argument("--experience-distilled", type=str, default="catalogs/semantic_experience_distilled_300.json")
    parser.add_argument("--output", "-o", type=str, default="catalogs/rules_unified.json")
    args = parser.parse_args()

    catalog = build_unified_catalog(
        knowledge_path=Path(args.knowledge),
        distilled_path=Path(args.experience_distilled),
        tagged_path=Path(args.experience_tagged),
    )
    output_path = Path(args.output)
    write_json(output_path, catalog)
    _print_summary(catalog, output_path)


if __name__ == "__main__":
    main()
