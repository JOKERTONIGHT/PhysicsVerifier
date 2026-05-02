from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from core.rule_catalog_retrieval import iter_rule_leaves, norm_text

from .builder import build_clusters, build_rule_tree, finalize_topics, make_rule_leaf
from .models import MaintenanceResult
from .normalization import normalize_topic, topic_key


def _find_topic(catalog: Dict[str, Any], domain: str, topic: str) -> Optional[Dict[str, Any]]:
    target_domain = norm_text(domain)
    target_topic = normalize_topic(target_domain, topic)
    for domain_obj in catalog.get("domains", []) or []:
        if norm_text(domain_obj.get("name") or "") != target_domain:
            continue
        for topic_obj in domain_obj.get("topics", []) or []:
            if norm_text(topic_obj.get("name") or "") == target_topic:
                return topic_obj
    return None


def _states_from_catalog(catalog: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    states: Dict[str, Dict[str, Any]] = {}
    for domain_obj in catalog.get("domains", []) or []:
        domain_name = norm_text(domain_obj.get("name") or "Unknown")
        for topic_obj in domain_obj.get("topics", []) or []:
            topic_name = norm_text(topic_obj.get("name") or "Unknown")
            states[topic_key(domain_name, topic_name)] = {
                "domain": domain_name,
                "topic": topic_name,
                "entry": topic_obj,
                "tagged_keyword_texts": list((topic_obj.get("tagged_reference") or {}).get("keywords") or []),
            }
    return states


def _ensure_topic(catalog: Dict[str, Any], domain: str, topic: str) -> Dict[str, Any]:
    domain_name = norm_text(domain) or "Unknown"
    topic_name = normalize_topic(domain_name, topic) or "Unknown"
    for domain_obj in catalog.setdefault("domains", []):
        if norm_text(domain_obj.get("name") or "") == domain_name:
            break
    else:
        domain_obj = {"name": domain_name, "topics": []}
        catalog["domains"].append(domain_obj)

    for topic_obj in domain_obj.setdefault("topics", []):
        if norm_text(topic_obj.get("name") or "") == topic_name:
            return topic_obj

    topic_obj = {
        "name": topic_name,
        "rules": [],
        "rule_tree": {"type": "topic", "label": topic_name, "children": []},
        "knowledge_reference": {"rule_ids": [], "keywords": []},
        "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
        "retrieval_hints": {"scene_keywords": [], "topic_keywords": [], "required_symbols": []},
        "clusters": [],
    }
    domain_obj["topics"].append(topic_obj)
    return topic_obj


def _recompute_metadata(catalog: Dict[str, Any]) -> None:
    domains = catalog.get("domains") or []
    total_topics = sum(len(domain.get("topics") or []) for domain in domains)
    total_rules = sum(len(list(iter_rule_leaves(topic))) for domain in domains for topic in domain.get("topics") or [])
    topics_with_rules = sum(1 for domain in domains for topic in domain.get("topics") or [] if list(iter_rule_leaves(topic)))
    total_clusters = sum(len(topic.get("clusters") or []) for domain in domains for topic in domain.get("topics") or [])
    meta = catalog.setdefault("metadata", {})
    meta.setdefault("version", "2.0")
    meta.setdefault("catalog_type", "unified_rules_v2")
    meta["total_domains"] = len(domains)
    meta["total_topics"] = total_topics
    meta["topics_with_rules"] = topics_with_rules
    meta["total_executable_rules"] = total_rules
    meta["total_clusters"] = total_clusters
    meta["hierarchy"] = ["domain", "topic", "context", "cluster", "rule"]


def add_experience_rules(catalog: Dict[str, Any], raw_rules: Iterable[Dict[str, Any]]) -> MaintenanceResult:
    updated = copy.deepcopy(catalog)
    existing: Set[str] = {
        str(rule.get("rule_id") or rule.get("id") or "")
        for domain in updated.get("domains", []) or []
        for topic in domain.get("topics", []) or []
        for rule in iter_rule_leaves(topic)
    }
    changed: List[str] = []
    warnings: List[str] = []

    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue
        domain = norm_text(raw_rule.get("domain") or "Unknown")
        topic = normalize_topic(domain, str(raw_rule.get("topic") or "Unknown"))
        rule_id = norm_text(raw_rule.get("rule_id") or raw_rule.get("id") or "")
        if not rule_id:
            warnings.append("skipped rule without rule_id")
            continue
        if rule_id in existing:
            warnings.append(f"skipped duplicate rule_id: {rule_id}")
            continue

        topic_obj = _ensure_topic(updated, domain, topic)
        topic_obj.setdefault("rules", []).append(make_rule_leaf(raw_rule, domain=domain, topic=topic))
        existing.add(rule_id)
        changed.append(rule_id)

    finalize_topics(_states_from_catalog(updated))
    _recompute_metadata(updated)
    return MaintenanceResult(catalog=updated, changed_rule_ids=changed, warnings=warnings)


def remove_rules(catalog: Dict[str, Any], rule_ids: Iterable[str]) -> MaintenanceResult:
    updated = copy.deepcopy(catalog)
    targets = {norm_text(rule_id) for rule_id in rule_ids if norm_text(rule_id)}
    removed: List[str] = []
    if not targets:
        return MaintenanceResult(catalog=updated, warnings=["no rule ids provided"])

    for domain in updated.get("domains", []) or []:
        for topic in domain.get("topics", []) or []:
            rules = [rule for rule in (topic.get("rules") or []) if isinstance(rule, dict)]
            kept = []
            for rule in rules:
                rid = norm_text(rule.get("rule_id") or rule.get("id") or "")
                if rid in targets:
                    removed.append(rid)
                else:
                    kept.append(rule)
            topic["rules"] = kept

    finalize_topics(_states_from_catalog(updated))
    _recompute_metadata(updated)
    return MaintenanceResult(catalog=updated, changed_rule_ids=removed)


def recluster_catalog(catalog: Dict[str, Any], *, domain: str | None = None, topic: str | None = None) -> MaintenanceResult:
    updated = copy.deepcopy(catalog)
    changed: List[str] = []
    for domain_obj in updated.get("domains", []) or []:
        domain_name = norm_text(domain_obj.get("name") or "")
        if domain and domain_name != norm_text(domain):
            continue
        for topic_obj in domain_obj.get("topics", []) or []:
            topic_name = norm_text(topic_obj.get("name") or "")
            if topic and topic_name != normalize_topic(domain_name, topic):
                continue
            rules = [rule for rule in (topic_obj.get("rules") or []) if isinstance(rule, dict)]
            topic_obj["clusters"] = build_clusters(rules)
            topic_obj["rule_tree"] = build_rule_tree(topic_name, rules, topic_obj["clusters"])
            changed.extend(str(rule.get("rule_id") or rule.get("id") or "") for rule in rules)

    _recompute_metadata(updated)
    return MaintenanceResult(catalog=updated, changed_rule_ids=[rid for rid in changed if rid])


def attach_symbolic_bindings(catalog: Dict[str, Any], manifest: Dict[str, Any]) -> MaintenanceResult:
    updated = copy.deepcopy(catalog)
    binding_by_rule: Dict[str, Dict[str, Any]] = {}
    for item in manifest.get("checks", []) if isinstance(manifest, dict) else []:
        if not isinstance(item, dict):
            continue
        rid = norm_text(item.get("rule_id") or "")
        if rid:
            binding_by_rule[rid] = {
                "function_name": item.get("function_name"),
                "source": item.get("source") or "experience_code_manifest",
            }

    changed: List[str] = []
    for domain in updated.get("domains", []) or []:
        for topic in domain.get("topics", []) or []:
            for rule in topic.get("rules", []) or []:
                if not isinstance(rule, dict):
                    continue
                rid = norm_text(rule.get("rule_id") or rule.get("id") or "")
                binding = binding_by_rule.get(rid)
                if binding:
                    rule["symbolic_binding"] = binding
                    changed.append(rid)

    return MaintenanceResult(catalog=updated, changed_rule_ids=changed)
