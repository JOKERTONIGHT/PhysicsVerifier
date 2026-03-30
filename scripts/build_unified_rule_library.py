from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def build_unified_library(rules_catalog: Dict[str, Any], distilled: Dict[str, Any]) -> Dict[str, Any]:
    topic_map: Dict[str, Dict[str, Any]] = {}

    for domain in rules_catalog.get("domains", []) or []:
        domain_name = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            topic_name = str(topic.get("name") or "Unknown")
            key = f"{domain_name}::{topic_name}"
            topic_map[key] = {
                "domain": domain_name,
                "topic": topic_name,
                "top_down_rules": [r for r in (topic.get("rules") or []) if isinstance(r, dict)],
                "experience_rules": [],
            }

    for er in distilled.get("rules", []) or []:
        if not isinstance(er, dict):
            continue
        domain_name = str(er.get("domain") or "Unknown")
        topic_name = str(er.get("topic") or "Unknown")
        key = f"{domain_name}::{topic_name}"
        bucket = topic_map.setdefault(
            key,
            {
                "domain": domain_name,
                "topic": topic_name,
                "top_down_rules": [],
                "experience_rules": [],
            },
        )
        bucket["experience_rules"].append(er)

    topics = list(topic_map.values())
    topics.sort(key=lambda x: (x["domain"], x["topic"]))

    return {
        "summary": {
            "topics": len(topics),
            "top_down_rules": sum(len(t["top_down_rules"]) for t in topics),
            "experience_rules": sum(len(t["experience_rules"]) for t in topics),
            "experience_source_summary": distilled.get("summary", {}),
        },
        "topics": topics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified rule library from top-down catalog and distilled experience rules.")
    parser.add_argument("--rules-catalog", type=str, default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--experience-distilled", type=str, required=True)
    parser.add_argument("--output", type=str, default="catalogs/unified_rule_library.json")
    args = parser.parse_args()

    catalog = _load_json(Path(args.rules_catalog))
    distilled = _load_json(Path(args.experience_distilled))
    unified = build_unified_library(catalog, distilled)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(unified, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Unified rule library saved to {out_path}")


if __name__ == "__main__":
    main()
