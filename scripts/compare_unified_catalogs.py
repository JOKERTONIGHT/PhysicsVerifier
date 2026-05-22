from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Set


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _console_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2)


def _topic_key(domain_name: str, topic_name: str) -> str:
    return f"{domain_name}::{topic_name}"


def _catalog_topics(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = str(topic.get("name") or "Unknown")
            rules = [rule for rule in (topic.get("rules") or []) if isinstance(rule, dict)]
            clusters = [cluster for cluster in (topic.get("scenario_clusters") or []) if isinstance(cluster, dict)]
            out[_topic_key(domain_name, topic_name)] = {
                "domain": domain_name,
                "topic": topic_name,
                "rule_ids": {str(rule.get("rule_id") or "") for rule in rules if str(rule.get("rule_id") or "")},
                "clusters": clusters,
            }
    return out


def _metadata_int(catalog: Dict[str, Any], key: str, fallback: int) -> int:
    meta = catalog.get("metadata") if isinstance(catalog.get("metadata"), dict) else {}
    try:
        return int(meta.get(key, fallback))
    except (TypeError, ValueError):
        return fallback


def _cluster_rule_ids(clusters: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for cluster in clusters:
        for rule_id in cluster.get("rule_ids") or []:
            text = str(rule_id or "")
            if text:
                out.add(text)
    return out


def _general_reasoning_rule_ids(clusters: List[Dict[str, Any]]) -> Set[str]:
    out: Set[str] = set()
    for cluster in clusters:
        cluster_id = str(cluster.get("id") or cluster.get("cluster_id") or "")
        if cluster_id != "general_reasoning":
            continue
        for rule_id in cluster.get("rule_ids") or []:
            text = str(rule_id or "")
            if text:
                out.add(text)
    return out


def compare_catalogs(baseline_path: Path, candidate_path: Path, output_path: Path | None = None) -> Dict[str, Any]:
    baseline = _load_json(baseline_path)
    candidate = _load_json(candidate_path)
    baseline_topics = _catalog_topics(baseline)
    candidate_topics = _catalog_topics(candidate)

    baseline_rule_total = _metadata_int(
        baseline,
        "total_executable_rules",
        sum(len(item["rule_ids"]) for item in baseline_topics.values()),
    )
    candidate_rule_total = _metadata_int(
        candidate,
        "total_executable_rules",
        sum(len(item["rule_ids"]) for item in candidate_topics.values()),
    )
    baseline_topics_with_rules = _metadata_int(
        baseline,
        "topics_with_rules",
        sum(1 for item in baseline_topics.values() if item["rule_ids"]),
    )
    candidate_topics_with_rules = _metadata_int(
        candidate,
        "topics_with_rules",
        sum(1 for item in candidate_topics.values() if item["rule_ids"]),
    )
    baseline_clusters = _metadata_int(
        baseline,
        "total_scenario_clusters",
        sum(len(item["clusters"]) for item in baseline_topics.values()),
    )
    candidate_clusters = _metadata_int(
        candidate,
        "total_scenario_clusters",
        sum(len(item["clusters"]) for item in candidate_topics.values()),
    )

    topic_rows: Dict[str, Dict[str, Any]] = {}
    for key in sorted(set(baseline_topics) | set(candidate_topics)):
        base = baseline_topics.get(key, {"rule_ids": set(), "clusters": []})
        cand = candidate_topics.get(key, {"rule_ids": set(), "clusters": []})
        candidate_rule_ids = cand["rule_ids"]
        covered_ids = _cluster_rule_ids(cand["clusters"]) & candidate_rule_ids
        general_ids = _general_reasoning_rule_ids(cand["clusters"]) & candidate_rule_ids
        candidate_rule_count = len(candidate_rule_ids)
        topic_rows[key] = {
            "baseline_rule_count": len(base["rule_ids"]),
            "candidate_rule_count": candidate_rule_count,
            "rule_delta": candidate_rule_count - len(base["rule_ids"]),
            "candidate_cluster_count": len(cand["clusters"]),
            "candidate_cluster_rule_coverage": (len(covered_ids) / candidate_rule_count) if candidate_rule_count else 0.0,
            "candidate_general_reasoning_rule_ratio": (len(general_ids) / candidate_rule_count) if candidate_rule_count else 0.0,
        }

    comparison = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "summary": {
            "baseline_total_rules": baseline_rule_total,
            "candidate_total_rules": candidate_rule_total,
            "rule_delta": candidate_rule_total - baseline_rule_total,
            "baseline_topics_with_rules": baseline_topics_with_rules,
            "candidate_topics_with_rules": candidate_topics_with_rules,
            "topics_with_rules_delta": candidate_topics_with_rules - baseline_topics_with_rules,
            "baseline_scenario_clusters": baseline_clusters,
            "candidate_scenario_clusters": candidate_clusters,
            "scenario_cluster_delta": candidate_clusters - baseline_clusters,
        },
        "topics": topic_rows,
    }
    if output_path:
        _write_json(output_path, comparison)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two unified rule catalogs.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    comparison = compare_catalogs(
        baseline_path=Path(args.baseline),
        candidate_path=Path(args.candidate),
        output_path=Path(args.output) if args.output else None,
    )
    print(_console_json(comparison))


if __name__ == "__main__":
    main()
