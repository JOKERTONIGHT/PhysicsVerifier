from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze_embedding_clusters(
    *,
    input_path: Path,
    output_path: Path | None = None,
    rule_input_path: Path | None = None,
    min_clustered_rule_ratio: float = 0.3,
    strict: bool = False,
) -> Dict[str, Any]:
    payload = _load_json(input_path)
    topics = [item for item in payload.get("topics", []) if isinstance(item, dict)]

    topic_reports: List[Dict[str, Any]] = []
    total_rules = 0
    total_clustered_rules = 0
    total_residual_rules = 0
    topics_with_clusters = 0
    high_residual_topics: List[str] = []
    invalid_topic_keys: List[str] = []
    seen_topic_keys: set[str] = set()
    duplicate_topic_keys: List[str] = []
    assignments_by_topic: Dict[str, set[str]] = {}

    for topic in topics:
        clusters = [item for item in topic.get("clusters", []) if isinstance(item, dict)]
        residual_ids = [str(item) for item in (topic.get("residual_rule_ids") or []) if str(item).strip()]
        rule_count = int(topic.get("rule_count") or 0)
        clustered_rules = sum(len(cluster.get("rule_ids") or []) for cluster in clusters)
        residual_rules = len(residual_ids)
        clustered_ratio = clustered_rules / rule_count if rule_count else 0.0
        residual_ratio = residual_rules / rule_count if rule_count else 0.0
        topic_key = str(topic.get("topic_key") or "")
        cluster_ids = [str(cluster.get("cluster_id") or "") for cluster in clusters]
        assigned_ids = [
            str(rule_id)
            for cluster in clusters
            for rule_id in (cluster.get("rule_ids") or [])
            if str(rule_id).strip()
        ] + residual_ids
        structurally_valid = bool(topic_key) and (
            len(cluster_ids) == len(set(cluster_ids))
            and all(cluster_ids)
            and len(assigned_ids) == len(set(assigned_ids))
            and len(assigned_ids) == rule_count
        )
        if not structurally_valid:
            invalid_topic_keys.append(topic_key)
        normalized_topic_key = topic_key.casefold()
        assignments_by_topic[normalized_topic_key] = set(assigned_ids)
        if normalized_topic_key in seen_topic_keys:
            duplicate_topic_keys.append(topic_key)
        seen_topic_keys.add(normalized_topic_key)

        total_rules += rule_count
        total_clustered_rules += clustered_rules
        total_residual_rules += residual_rules
        if clusters:
            topics_with_clusters += 1
        if rule_count >= 20 and residual_ratio > 0.7:
            high_residual_topics.append(topic_key)

        topic_reports.append(
            {
                "topic_key": topic_key,
                "rule_count": rule_count,
                "cluster_count": len(clusters),
                "clustered_rule_count": clustered_rules,
                "residual_rule_count": residual_rules,
                "clustered_rule_ratio": round(clustered_ratio, 4),
                "residual_rule_ratio": round(residual_ratio, 4),
                "largest_cluster_size": max((len(cluster.get("rule_ids") or []) for cluster in clusters), default=0),
                "structurally_valid": structurally_valid,
            }
        )

    topic_reports.sort(key=lambda item: (-item["rule_count"], item["topic_key"]))
    clustered_rule_ratio = total_clustered_rules / total_rules if total_rules else 0.0
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata_rule_count = int(metadata.get("rule_count") or total_rules)
    metadata_topic_count = int(metadata.get("topic_count") or len(topics))
    source_alignment_mismatches: List[str] = []
    if rule_input_path:
        rule_input = _load_json(rule_input_path)
        expected_by_topic: Dict[str, set[str]] = {}
        for rule in rule_input.get("rules", []) or []:
            if not isinstance(rule, dict):
                continue
            topic_key = str(rule.get("topic_key") or "").casefold()
            rule_id = str(rule.get("rule_id") or "")
            if topic_key and rule_id:
                expected_by_topic.setdefault(topic_key, set()).add(rule_id)
        for topic_key in sorted(set(expected_by_topic) | set(assignments_by_topic)):
            if expected_by_topic.get(topic_key, set()) != assignments_by_topic.get(topic_key, set()):
                source_alignment_mismatches.append(topic_key)
    source_alignment_valid = not source_alignment_mismatches
    structural_valid = (
        not invalid_topic_keys
        and not duplicate_topic_keys
        and metadata_rule_count == total_rules
        and metadata_topic_count == len(topics)
        and source_alignment_valid
    )
    report = {
        "input": str(input_path),
        "metadata": metadata,
        "topic_count": len(topics),
        "topics_with_clusters": topics_with_clusters,
        "total_rule_count": total_rules,
        "total_clustered_rule_count": total_clustered_rules,
        "total_residual_rule_count": total_residual_rules,
        "clustered_rule_ratio": round(clustered_rule_ratio, 4),
        "high_residual_topics": high_residual_topics,
        "structural_valid": structural_valid,
        "source_alignment_valid": source_alignment_valid,
        "source_alignment_mismatches": source_alignment_mismatches,
        "invalid_topic_keys": invalid_topic_keys,
        "duplicate_topic_keys": duplicate_topic_keys,
        "metadata_count_mismatches": {
            **(
                {"rule_count": {"metadata": metadata_rule_count, "actual": total_rules}}
                if metadata_rule_count != total_rules
                else {}
            ),
            **(
                {"topic_count": {"metadata": metadata_topic_count, "actual": len(topics)}}
                if metadata_topic_count != len(topics)
                else {}
            ),
        },
        "topics": topic_reports,
        "ready_for_labeling": (
            bool(topics)
            and structural_valid
            and clustered_rule_ratio >= min_clustered_rule_ratio
        ),
        "min_clustered_rule_ratio": min_clustered_rule_ratio,
    }

    if output_path:
        _write_json(output_path, report)
    if strict and not report["ready_for_labeling"]:
        raise SystemExit(1)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze topic-local rule embedding clustering output.")
    parser.add_argument("--input", default="results/unified_rules_3000/rule_embedding_clusters.json")
    parser.add_argument("--output", default="results/unified_rules_3000/rule_embedding_cluster_report.json")
    parser.add_argument("--rule-input", default="")
    parser.add_argument("--min-clustered-rule-ratio", type=float, default=0.3)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = analyze_embedding_clusters(
        input_path=Path(args.input),
        output_path=Path(args.output),
        rule_input_path=Path(args.rule_input) if args.rule_input else None,
        min_clustered_rule_ratio=float(args.min_clustered_rule_ratio),
        strict=bool(args.strict),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "topics"}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
