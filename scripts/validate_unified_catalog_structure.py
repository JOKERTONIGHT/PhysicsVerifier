from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _text(value: Any) -> str:
    return str(value or "").strip()


def _ids(values: Iterable[Any]) -> List[str]:
    return [_text(value) for value in values if _text(value)]


def validate_catalog_structure(catalog: Dict[str, Any]) -> Dict[str, Any]:
    """Validate catalog-wide identity, reference, and tree-reachability invariants."""
    domains = [item for item in (catalog.get("domains") or []) if isinstance(item, dict)]
    topics: List[Dict[str, Any]] = []
    all_rule_ids: List[str] = []
    duplicate_rule_ids_within_topic: List[str] = []
    duplicate_cluster_ids: List[str] = []
    empty_rule_ids: List[str] = []
    empty_cluster_ids: List[str] = []
    topics_without_clusters: List[str] = []
    topics_with_unknown_cluster_rules: List[str] = []
    topics_with_duplicate_cluster_assignments: List[str] = []
    topics_with_unreachable_rules: List[str] = []
    clusters_with_invalid_groups: List[str] = []

    for domain in domains:
        domain_name = _text(domain.get("name") or domain.get("id") or "unknown")
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topics.append(topic)
            topic_name = _text(topic.get("name") or topic.get("id") or "unknown")
            topic_key = f"{domain_name}::{topic_name}"
            rules = [item for item in (topic.get("rules") or []) if isinstance(item, dict)]
            raw_rule_ids = [_text(rule.get("rule_id")) for rule in rules]
            if any(not rule_id for rule_id in raw_rule_ids):
                empty_rule_ids.append(topic_key)
            rule_ids = [rule_id for rule_id in raw_rule_ids if rule_id]
            all_rule_ids.extend(rule_ids)
            if len(rule_ids) != len(set(rule_ids)):
                duplicate_rule_ids_within_topic.append(topic_key)

            clusters = [item for item in (topic.get("scenario_clusters") or []) if isinstance(item, dict)]
            if rule_ids and not clusters:
                topics_without_clusters.append(topic_key)
            raw_cluster_ids = [_text(cluster.get("id") or cluster.get("cluster_id")) for cluster in clusters]
            if any(not cluster_id for cluster_id in raw_cluster_ids):
                empty_cluster_ids.append(topic_key)
            cluster_ids = [cluster_id for cluster_id in raw_cluster_ids if cluster_id]
            if len(cluster_ids) != len(set(cluster_ids)):
                duplicate_cluster_ids.append(topic_key)

            assigned_ids: List[str] = []
            for cluster_index, cluster in enumerate(clusters):
                cluster_key = f"{topic_key}::{raw_cluster_ids[cluster_index] or cluster_index}"
                cluster_rule_ids = _ids(cluster.get("rule_ids") or [])
                assigned_ids.extend(cluster_rule_ids)
                groups = [item for item in (cluster.get("rule_groups") or []) if isinstance(item, dict)]
                if groups:
                    group_rule_ids: List[str] = []
                    for group in groups:
                        group_rule_ids.extend(_ids(group.get("rule_ids") or []))
                    if (
                        len(group_rule_ids) != len(set(group_rule_ids))
                        or set(group_rule_ids) != set(cluster_rule_ids)
                    ):
                        clusters_with_invalid_groups.append(cluster_key)

            rule_id_set = set(rule_ids)
            assigned_set = set(assigned_ids)
            if not assigned_set.issubset(rule_id_set):
                topics_with_unknown_cluster_rules.append(topic_key)
            if len(assigned_ids) != len(assigned_set):
                topics_with_duplicate_cluster_assignments.append(topic_key)
            if assigned_set != rule_id_set:
                topics_with_unreachable_rules.append(topic_key)

    duplicate_global_rule_ids = sorted(
        rule_id for rule_id, count in Counter(all_rule_ids).items() if count > 1
    )
    metadata = catalog.get("metadata") if isinstance(catalog.get("metadata"), dict) else {}
    actual_cluster_count = sum(
        len([item for item in (topic.get("scenario_clusters") or []) if isinstance(item, dict)])
        for topic in topics
    )
    topics_with_rules = sum(
        1
        for topic in topics
        if any(
            isinstance(rule, dict) and _text(rule.get("rule_id"))
            for rule in (topic.get("rules") or [])
        )
    )
    metadata_mismatches: Dict[str, Dict[str, int]] = {}
    expected_counts = {
        "total_domains": len(domains),
        "total_topics": len(topics),
        "topics_with_rules": topics_with_rules,
        "total_executable_rules": len(all_rule_ids),
        "total_scenario_clusters": actual_cluster_count,
    }
    for key, actual in expected_counts.items():
        if key in metadata and int(metadata.get(key) or 0) != actual:
            metadata_mismatches[key] = {
                "metadata": int(metadata.get(key) or 0),
                "actual": actual,
            }

    errors = {
        "duplicate_global_rule_ids": duplicate_global_rule_ids,
        "duplicate_rule_ids_within_topic": sorted(set(duplicate_rule_ids_within_topic)),
        "duplicate_cluster_ids": sorted(set(duplicate_cluster_ids)),
        "empty_rule_ids": sorted(set(empty_rule_ids)),
        "empty_cluster_ids": sorted(set(empty_cluster_ids)),
        "topics_without_clusters": sorted(set(topics_without_clusters)),
        "topics_with_unknown_cluster_rules": sorted(set(topics_with_unknown_cluster_rules)),
        "topics_with_duplicate_cluster_assignments": sorted(
            set(topics_with_duplicate_cluster_assignments)
        ),
        "topics_with_unreachable_rules": sorted(set(topics_with_unreachable_rules)),
        "clusters_with_invalid_groups": sorted(set(clusters_with_invalid_groups)),
        "metadata_mismatches": metadata_mismatches,
    }
    return {
        "valid": not any(errors.values()),
        "counts": {
            "domains": len(domains),
            "topics": len(topics),
            "topics_with_rules": topics_with_rules,
            "rules": len(all_rule_ids),
            "scenario_clusters": actual_cluster_count,
        },
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate unified catalog identity and rule-tree reachability."
    )
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise RuntimeError("Catalog payload must be a JSON object.")
    report = validate_catalog_structure(catalog)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.fail_on_invalid and not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
