from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _ordered_unique(items: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        text = _norm_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _topic_key(domain: str, topic: str) -> str:
    return f"{_norm_text(domain).casefold()}::{_norm_text(topic).casefold()}"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _topic_rule_ids(topic: Dict[str, Any]) -> List[str]:
    return [
        _norm_text(rule.get("rule_id") or "")
        for rule in (topic.get("rules") or [])
        if isinstance(rule, dict) and _norm_text(rule.get("rule_id") or "")
    ]


def _collect_blueprint_rule_ids(cluster_defs: List[Dict[str, Any]]) -> List[str]:
    rule_ids: List[str] = []
    for cluster in cluster_defs or []:
        if not isinstance(cluster, dict):
            continue
        group_rule_ids: List[str] = []
        for group in cluster.get("rule_groups", []) or []:
            if not isinstance(group, dict):
                continue
            group_rule_ids.extend(
                [
                    _norm_text(rule_id)
                    for rule_id in (group.get("rule_ids") or [])
                    if _norm_text(rule_id)
                ]
            )
        if group_rule_ids:
            rule_ids.extend(group_rule_ids)
            continue
        rule_ids.extend(
            [
                _norm_text(rule_id)
                for rule_id in (cluster.get("rule_ids") or [])
                if _norm_text(rule_id)
            ]
        )
    return rule_ids


def validate_blueprints_against_catalog(
    catalog: Dict[str, Any],
    blueprints: Dict[str, List[Dict[str, Any]]],
    *,
    mode: str = "full",
) -> Dict[str, Any]:
    mode = _norm_text(mode).casefold() or "full"
    if mode not in {"full", "subset"}:
        raise ValueError("mode must be either 'full' or 'subset'.")

    topic_rule_index: Dict[str, List[str]] = {}
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = _norm_text(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            rule_ids = _topic_rule_ids(topic)
            if not rule_ids:
                continue
            topic_rule_index[_topic_key(domain_name, _norm_text(topic.get("name") or "Unknown"))] = rule_ids

    missing_topics: List[str] = []
    topics_with_empty_clusters: List[str] = []
    topics_with_duplicate_rule_assignments: List[str] = []
    topics_with_unknown_rules: List[str] = []
    topics_with_uncovered_rules: List[str] = []
    unknown_blueprint_topics: List[str] = []

    if mode == "full":
        topics_to_validate = sorted(topic_rule_index.keys())
    else:
        topics_to_validate = []
        for raw_key in blueprints.keys():
            topic_key = _norm_text(raw_key).casefold()
            if not topic_key:
                continue
            if topic_key not in topic_rule_index:
                unknown_blueprint_topics.append(topic_key)
                continue
            topics_to_validate.append(topic_key)
        topics_to_validate = sorted(_ordered_unique(topics_to_validate))

    for topic_key in topics_to_validate:
        rule_ids = topic_rule_index[topic_key]
        cluster_defs = blueprints.get(topic_key)
        if not cluster_defs:
            missing_topics.append(topic_key)
            continue
        if not any(isinstance(cluster, dict) for cluster in cluster_defs):
            topics_with_empty_clusters.append(topic_key)
            continue
        assigned_rule_ids = _collect_blueprint_rule_ids(cluster_defs)
        if not assigned_rule_ids:
            topics_with_empty_clusters.append(topic_key)
            continue
        assigned_set = set(assigned_rule_ids)
        topic_rule_set = set(rule_ids)
        if len(assigned_rule_ids) != len(assigned_set):
            topics_with_duplicate_rule_assignments.append(topic_key)
        if not assigned_set.issubset(topic_rule_set):
            topics_with_unknown_rules.append(topic_key)
        if assigned_set != topic_rule_set:
                topics_with_uncovered_rules.append(topic_key)

    report = {
        "valid": not any(
            [
                missing_topics,
                topics_with_empty_clusters,
                topics_with_duplicate_rule_assignments,
                topics_with_unknown_rules,
                topics_with_uncovered_rules,
                unknown_blueprint_topics,
            ]
        ),
        "mode": mode,
        "topic_count_with_rules": len(topic_rule_index),
        "blueprint_topic_count": len([key for key in blueprints.keys() if _norm_text(key)]),
        "validated_topic_count": len(topics_to_validate),
        "missing_topics": sorted(missing_topics),
        "unknown_blueprint_topics": sorted(unknown_blueprint_topics),
        "topics_with_empty_clusters": sorted(topics_with_empty_clusters),
        "topics_with_duplicate_rule_assignments": sorted(topics_with_duplicate_rule_assignments),
        "topics_with_unknown_rules": sorted(topics_with_unknown_rules),
        "topics_with_uncovered_rules": sorted(topics_with_uncovered_rules),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate generated scenario-cluster blueprints against a unified catalog.")
    parser.add_argument("--catalog", type=str, required=True)
    parser.add_argument("--blueprints", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--mode", choices=["full", "subset"], default="full")
    parser.add_argument("--fail-on-invalid", action="store_true")
    args = parser.parse_args()

    catalog = _load_json(Path(args.catalog))
    blueprints = _load_json(Path(args.blueprints))
    if not isinstance(blueprints, dict):
        raise RuntimeError("Blueprint payload must be a dict keyed by topic_key.")
    report = validate_blueprints_against_catalog(catalog, blueprints, mode=args.mode)
    if args.output:
        _dump_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_invalid and not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
