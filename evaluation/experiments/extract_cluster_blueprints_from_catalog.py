#!/usr/bin/env python3
"""Extract scenario-cluster blueprints from a unified rules catalog (e.g. rules_unified_3000)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_unified_catalog import _topic_key  # noqa: E402


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _ordered_unique(values: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = _norm_text(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def extract_blueprints(catalog: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = _norm_text(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = _norm_text(topic.get("name") or "Unknown")
            topic_key = _topic_key(domain_name, topic_name).casefold()
            clusters: List[Dict[str, Any]] = []
            for cluster in topic.get("scenario_clusters", []) or []:
                if not isinstance(cluster, dict):
                    continue
                rule_groups: List[Dict[str, Any]] = []
                for group in cluster.get("rule_groups", []) or []:
                    if not isinstance(group, dict):
                        continue
                    rule_ids = _ordered_unique([str(x) for x in (group.get("rule_ids") or [])])
                    if not rule_ids:
                        continue
                    rule_groups.append(
                        {
                            "group_id": _norm_text(group.get("id") or group.get("group_id") or ""),
                            "name": _norm_text(group.get("name") or ""),
                            "summary": _norm_text(group.get("summary") or ""),
                            "activation_condition": _norm_text(group.get("activation_condition") or group.get("summary") or ""),
                            "rule_ids": rule_ids,
                        }
                    )
                cluster_rule_ids = _ordered_unique([str(x) for x in (cluster.get("rule_ids") or [])])
                if not rule_groups and cluster_rule_ids:
                    rule_groups.append(
                        {
                            "group_id": f"{_norm_text(cluster.get('id') or 'cluster')}_rules",
                            "name": _norm_text(cluster.get("name") or "Cluster Rules"),
                            "summary": _norm_text(cluster.get("summary") or ""),
                            "activation_condition": _norm_text(cluster.get("summary") or ""),
                            "rule_ids": cluster_rule_ids,
                        }
                    )
                if not rule_groups:
                    continue
                clusters.append(
                    {
                        "cluster_id": _norm_text(cluster.get("id") or ""),
                        "name": _norm_text(cluster.get("name") or ""),
                        "description": _norm_text(cluster.get("summary") or cluster.get("description") or ""),
                        "includes": [],
                        "excludes": [],
                        "entry_cues": [],
                        "related_clusters": [],
                        "rule_groups": rule_groups,
                    }
                )
            if clusters:
                out[topic_key] = clusters
    return out


def filter_blueprints(
    blueprints: Dict[str, List[Dict[str, Any]]],
    allowed_rule_ids: set[str],
) -> Dict[str, List[Dict[str, Any]]]:
    filtered: Dict[str, List[Dict[str, Any]]] = {}
    for topic_key, clusters in blueprints.items():
        kept_clusters: List[Dict[str, Any]] = []
        for cluster in clusters:
            groups: List[Dict[str, Any]] = []
            cluster_rule_ids: List[str] = []
            for group in cluster.get("rule_groups", []) or []:
                rule_ids = [rid for rid in (group.get("rule_ids") or []) if rid in allowed_rule_ids]
                if not rule_ids:
                    continue
                group_copy = dict(group)
                group_copy["rule_ids"] = rule_ids
                groups.append(group_copy)
                cluster_rule_ids.extend(rule_ids)
            if not groups:
                continue
            cluster_copy = dict(cluster)
            cluster_copy["rule_groups"] = groups
            cluster_copy["rule_ids"] = _ordered_unique(cluster_rule_ids)
            kept_clusters.append(cluster_copy)
        if kept_clusters:
            filtered[topic_key] = kept_clusters
    return filtered


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract/filter scenario cluster blueprints from unified catalog.")
    parser.add_argument("--catalog", type=str, default="catalogs/rules_unified_3000.json")
    parser.add_argument("--output", type=str, default="catalogs/cluster_blueprints_from_unified_3000.json")
    parser.add_argument("--allowed-rules-json", type=str, default="", help="Optional JSON list of allowed norm_ rule ids.")
    args = parser.parse_args()

    catalog = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
    blueprints = extract_blueprints(catalog)
    if args.allowed_rules_json:
        allowed = set(json.loads(Path(args.allowed_rules_json).read_text(encoding="utf-8")))
        blueprints = filter_blueprints(blueprints, allowed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blueprints, ensure_ascii=False, indent=2), encoding="utf-8")
    cluster_count = sum(len(items) for items in blueprints.values())
    print(json.dumps({"output": str(out_path), "topic_count": len(blueprints), "cluster_count": cluster_count}, indent=2))


if __name__ == "__main__":
    main()
