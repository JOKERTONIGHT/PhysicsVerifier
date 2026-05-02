from __future__ import annotations

import datetime as _dt
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from core.rule_catalog_retrieval import (
    apply_manual_rule_override,
    apply_manual_topic_hint_override,
    build_scene_keywords,
    build_topic_required_symbols,
    classify_rule_scope,
    extract_keywords,
    norm_text,
    ordered_unique,
)

from .io import load_json
from .models import RulePath
from .normalization import (
    build_match_features,
    normalize_topic,
    ordered_strings,
    safe_symbolic_hint,
    slugify,
    topic_key,
)

CLUSTER_TOPIC_THRESHOLD = 12
CLUSTER_BUCKET_THRESHOLD = 3


def _cluster_keywords(rules: List[Dict[str, Any]], *, max_keywords: int = 8) -> List[str]:
    counter: Counter[str] = Counter()
    display: Dict[str, str] = {}
    order: Dict[str, int] = {}

    for rule in rules:
        features = rule.get("match_features") if isinstance(rule.get("match_features"), dict) else {}
        tokens = list(features.get("trigger_keywords") or []) + list(features.get("object_keywords") or [])
        for token in tokens:
            item = norm_text(token)
            if not item:
                continue
            key = item.casefold()
            counter[key] += 1
            display.setdefault(key, item)
            order.setdefault(key, len(order))

    ranked = sorted(counter.items(), key=lambda item: (-item[1], order[item[0]], item[0]))
    return [display[key] for key, _ in ranked[:max_keywords]]


def build_clusters(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(rules) < CLUSTER_TOPIC_THRESHOLD:
        return []

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for rule in rules:
        error_type = norm_text(rule.get("error_type") or "unknown") or "unknown"
        buckets.setdefault(error_type, []).append(rule)

    clusters: List[Dict[str, Any]] = []
    for error_type, members in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(members) < CLUSTER_BUCKET_THRESHOLD:
            continue
        cluster_id = f"cluster_{slugify(error_type)}"
        clusters.append(
            {
                "cluster_id": cluster_id,
                "label": error_type,
                "error_type": error_type,
                "keywords": _cluster_keywords(members),
                "rule_ids": [str(rule["rule_id"]) for rule in members],
            }
        )
    return clusters


def build_rule_tree(topic_name: str, rules: List[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
    cluster_by_rule: Dict[str, str] = {}
    cluster_label_by_id: Dict[str, str] = {}
    for cluster in clusters:
        cluster_id = str(cluster.get("cluster_id") or "cluster_unknown")
        cluster_label_by_id[cluster_id] = str(cluster.get("label") or cluster_id)
        for rule_id in cluster.get("rule_ids") or []:
            cluster_by_rule[str(rule_id)] = cluster_id

    context_nodes: Dict[str, Dict[str, Any]] = {}
    for rule in rules:
        context = norm_text(rule.get("context") or rule.get("error_type") or "general") or "general"
        cluster_id = cluster_by_rule.get(str(rule.get("rule_id") or ""), "unclustered")
        context_node = context_nodes.setdefault(
            context,
            {"type": "context", "label": context, "children": {}},
        )
        cluster_node = context_node["children"].setdefault(
            cluster_id,
            {
                "type": "cluster",
                "cluster_id": cluster_id,
                "label": cluster_label_by_id.get(cluster_id, cluster_id),
                "rule_ids": [],
            },
        )
        cluster_node["rule_ids"].append(str(rule.get("rule_id") or ""))

    children = []
    for context, context_node in sorted(context_nodes.items(), key=lambda item: item[0]):
        cluster_children = list(context_node["children"].values())
        cluster_children.sort(key=lambda item: (item.get("cluster_id") == "unclustered", str(item.get("cluster_id") or "")))
        children.append({"type": "context", "label": context, "children": cluster_children})

    return {"type": "topic", "label": topic_name, "children": children}


def _build_topic_skeleton(knowledge_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    domains_out: List[Dict[str, Any]] = []
    states: Dict[str, Dict[str, Any]] = {}

    for domain in knowledge_data.get("domains", []) or []:
        domain_name = norm_text(domain.get("name") or "Unknown")
        domain_out = {"name": domain_name, "topics": []}

        for topic in domain.get("topics", []) or []:
            topic_name = norm_text(topic.get("name") or "Unknown")
            knowledge_rules = [item for item in (topic.get("rules") or []) if isinstance(item, dict)]
            knowledge_rule_ids = ordered_strings(item.get("id") or "" for item in knowledge_rules)
            knowledge_texts: List[str] = [topic_name]
            for item in knowledge_rules:
                knowledge_texts.extend(
                    [
                        str(item.get("title") or ""),
                        str(item.get("description") or ""),
                        str(item.get("check_logic") or ""),
                    ]
                )

            entry = {
                "name": topic_name,
                "rules": [],
                "rule_tree": {"type": "topic", "label": topic_name, "children": []},
                "knowledge_reference": {
                    "rule_ids": knowledge_rule_ids,
                    "keywords": extract_keywords(knowledge_texts, max_keywords=16),
                },
                "tagged_reference": {
                    "source_ids": [],
                    "titles": [],
                    "aliases": [],
                    "keywords": [],
                },
                "retrieval_hints": {
                    "topic_keywords": [],
                    "required_symbols": [],
                },
                "clusters": [],
            }
            domain_out["topics"].append(entry)

            key = topic_key(domain_name, topic_name)
            states[key] = {
                "domain": domain_name,
                "topic": topic_name,
                "entry": entry,
                "tagged_keyword_texts": [],
            }

        domains_out.append(domain_out)

    return domains_out, states


def make_rule_leaf(raw_rule: Dict[str, Any], *, domain: str, topic: str) -> Dict[str, Any]:
    symbolic_hint = safe_symbolic_hint(raw_rule.get("symbolic_hint"))
    title = norm_text(raw_rule.get("title") or "")
    trigger = norm_text(raw_rule.get("trigger") or "")
    check_logic = norm_text(raw_rule.get("check_logic") or "")
    error_type = norm_text(raw_rule.get("error_type") or "logic") or "logic"
    rule_id = norm_text(raw_rule.get("rule_id") or raw_rule.get("id") or "")
    path = RulePath(domain=domain, topic=topic, context=norm_text(raw_rule.get("context") or error_type) or "general")
    rule_leaf = {
        "id": rule_id,
        "rule_id": rule_id,
        "title": title,
        "trigger": trigger,
        "check_logic": check_logic,
        "error_type": error_type,
        "context": path.context,
        "path": path.as_dict(),
        "scope": classify_rule_scope(
            title=title,
            trigger=trigger,
            check_logic=check_logic,
            rule_id=rule_id,
        ),
        "symbolic_hint": symbolic_hint,
        "symbolic_binding": dict(raw_rule.get("symbolic_binding") or {}),
        "support": {
            "count": int(raw_rule.get("count") or (raw_rule.get("support") or {}).get("count") or 0),
            "sample_ids": ordered_strings(
                raw_rule.get("sample_ids") or (raw_rule.get("support") or {}).get("sample_ids") or []
            ),
        },
        "match_features": build_match_features(title, trigger, check_logic, symbolic_hint),
    }
    return apply_manual_rule_override(rule_leaf)


def attach_distilled_rules(states: Dict[str, Dict[str, Any]], distilled_data: Dict[str, Any]) -> None:
    raw_rules = distilled_data.get("rules") if isinstance(distilled_data, dict) else []
    if not isinstance(raw_rules, list):
        raise ValueError("Distilled experience data must contain a top-level 'rules' list.")

    unmatched: List[str] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue

        domain = norm_text(raw_rule.get("domain") or "Unknown")
        topic = normalize_topic(domain, str(raw_rule.get("topic") or "Unknown"))
        key = topic_key(domain, topic)
        state = states.get(key)
        if state is None:
            unmatched.append(f"{domain} / {topic} :: {raw_rule.get('rule_id')}")
            continue

        state["entry"]["rules"].append(make_rule_leaf(raw_rule, domain=domain, topic=topic))

    if unmatched:
        uniq = ordered_unique(unmatched)
        preview = "\n".join(f"- {item}" for item in uniq[:10])
        raise ValueError(
            "Distilled rules contain topics that do not map to the knowledge skeleton.\n"
            f"Unmatched unique topics/rules: {len(uniq)}\n{preview}"
        )


def attach_tagged_reference(states: Dict[str, Dict[str, Any]], tagged_data: Any) -> int:
    if not isinstance(tagged_data, list):
        raise ValueError("Tagged experience data must be a JSON list.")

    mapped_rules = 0
    for raw_rule in tagged_data:
        if not isinstance(raw_rule, dict):
            continue
        tags = raw_rule.get("tags") if isinstance(raw_rule.get("tags"), dict) else {}
        domain = norm_text(tags.get("domain") or "Unknown")
        topic = normalize_topic(domain, str(tags.get("topic") or "Unknown"))
        state = states.get(topic_key(domain, topic))
        if state is None:
            continue

        title = norm_text(raw_rule.get("title") or "")
        description = norm_text(raw_rule.get("description") or "")
        source_id = norm_text(raw_rule.get("id") or "")

        ref = state["entry"]["tagged_reference"]
        ref["source_ids"].append(source_id)
        ref["titles"].append(title)
        ref["aliases"].append(title)
        state["tagged_keyword_texts"].extend([title, description])
        mapped_rules += 1

    for state in states.values():
        ref = state["entry"]["tagged_reference"]
        ref["source_ids"] = ordered_unique(ref["source_ids"])
        ref["titles"] = ordered_unique(ref["titles"])
        ref["aliases"] = ordered_unique(ref["aliases"])
        ref["keywords"] = extract_keywords(state["tagged_keyword_texts"], max_keywords=16)

    return mapped_rules


def finalize_topics(states: Dict[str, Dict[str, Any]]) -> None:
    for state in states.values():
        entry = state["entry"]
        rules = entry["rules"]
        rules.sort(
            key=lambda item: (
                -int((item.get("support") or {}).get("count", 0)),
                str(item.get("rule_id") or ""),
            )
        )

        domain_rule_texts: List[str] = []
        for rule in rules:
            if norm_text(rule.get("scope") or "domain") != "domain":
                continue
            domain_rule_texts.extend([norm_text(rule.get("title") or ""), norm_text(rule.get("trigger") or "")])

        topic_keywords = ordered_unique(
            list(extract_keywords([state["topic"]], max_keywords=8))
            + list(entry["knowledge_reference"].get("keywords") or [])
            + list(entry["tagged_reference"].get("keywords") or [])
        )[:20]
        scene_keywords = build_scene_keywords(
            topic_name=state["topic"],
            tagged_titles=entry["tagged_reference"].get("titles") or [],
            tagged_aliases=entry["tagged_reference"].get("aliases") or [],
            rule_texts=domain_rule_texts,
        )
        scene_keywords, topic_keywords = apply_manual_topic_hint_override(
            domain=state["domain"],
            topic=state["topic"],
            scene_keywords=scene_keywords,
            topic_keywords=topic_keywords,
        )

        entry["retrieval_hints"] = {
            "scene_keywords": scene_keywords,
            "topic_keywords": topic_keywords,
            "required_symbols": build_topic_required_symbols(rules),
        }
        entry["clusters"] = build_clusters(rules)
        entry["rule_tree"] = build_rule_tree(state["topic"], rules, entry["clusters"])

        cluster_by_rule: Dict[str, str] = {}
        for cluster in entry["clusters"]:
            for rule_id in cluster.get("rule_ids") or []:
                cluster_by_rule[str(rule_id)] = str(cluster.get("cluster_id") or "unclustered")
        for rule in rules:
            path = dict(rule.get("path") or {})
            path["cluster"] = cluster_by_rule.get(str(rule.get("rule_id") or ""), "unclustered")
            rule["path"] = path


def build_unified_catalog_from_data(
    knowledge_data: Dict[str, Any],
    distilled_data: Dict[str, Any],
    tagged_data: Any,
) -> Dict[str, Any]:
    domains_out, states = _build_topic_skeleton(knowledge_data)
    attach_distilled_rules(states, distilled_data)
    mapped_tagged_rules = attach_tagged_reference(states, tagged_data)
    finalize_topics(states)

    total_topics = sum(len(domain["topics"]) for domain in domains_out)
    total_rules = sum(len(topic["rules"]) for domain in domains_out for topic in domain["topics"])
    topics_with_rules = sum(1 for domain in domains_out for topic in domain["topics"] if topic["rules"])
    total_clusters = sum(len(topic["clusters"]) for domain in domains_out for topic in domain["topics"])
    clustered_topics = sum(1 for domain in domains_out for topic in domain["topics"] if topic["clusters"])
    knowledge_rule_references = sum(
        len(topic["knowledge_reference"]["rule_ids"]) for domain in domains_out for topic in domain["topics"]
    )

    return {
        "metadata": {
            "version": "2.0",
            "catalog_type": "unified_rules_v2",
            "generated_at": _dt.datetime.now().isoformat(),
            "total_domains": len(domains_out),
            "total_topics": total_topics,
            "topics_with_rules": topics_with_rules,
            "total_executable_rules": total_rules,
            "knowledge_rule_references": knowledge_rule_references,
            "mapped_tagged_reference_rules": mapped_tagged_rules,
            "clustered_topics": clustered_topics,
            "total_clusters": total_clusters,
            "cluster_topic_threshold": CLUSTER_TOPIC_THRESHOLD,
            "cluster_bucket_threshold": CLUSTER_BUCKET_THRESHOLD,
            "hierarchy": ["domain", "topic", "context", "cluster", "rule"],
        },
        "domains": domains_out,
    }


def build_unified_catalog(knowledge_path: Path, distilled_path: Path, tagged_path: Path) -> Dict[str, Any]:
    return build_unified_catalog_from_data(load_json(knowledge_path), load_json(distilled_path), load_json(tagged_path))


def _experience_rule_to_catalog_rule(er: Dict[str, Any]) -> Dict[str, Any]:
    rid = str(er.get("rule_id") or "")
    title = str(er.get("title") or rid)
    trigger = str(er.get("trigger") or "").strip()
    logic = str(er.get("check_logic") or "").strip()
    desc_parts: List[str] = []
    if trigger:
        desc_parts.append(f"Trigger: {trigger}")
    if logic:
        desc_parts.append(f"Check Logic: {logic}")
    description = "\n".join(desc_parts) if desc_parts else logic
    out: Dict[str, Any] = {
        "id": rid,
        "title": title,
        "description": description,
        "check_logic": logic,
        "source": "experience",
        "error_type": str(er.get("error_type") or ""),
    }
    if trigger:
        out["trigger"] = trigger
    hint = er.get("symbolic_hint") if isinstance(er.get("symbolic_hint"), dict) else None
    if hint:
        out["symbolic_hint"] = hint
    return out


def build_simple_unified_library(
    rules_catalog: Dict[str, Any],
    distilled: Dict[str, Any],
    *,
    rule_source: str = "hybrid",
) -> Dict[str, Any]:
    """Compatibility builder for the older flat unified catalog format."""
    topic_map: Dict[str, Dict[str, Any]] = {}
    for domain in rules_catalog.get("domains", []) or []:
        domain_name = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            topic_name = normalize_topic(domain_name, str(topic.get("name") or "Unknown"))
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
        topic_name = normalize_topic(domain_name, str(er.get("topic") or "Unknown"))
        bucket = topic_map.setdefault(
            f"{domain_name}::{topic_name}",
            {"domain": domain_name, "topic": topic_name, "top_down_rules": [], "experience_rules": []},
        )
        bucket["experience_rules"].append(_experience_rule_to_catalog_rule(er))

    domains_out: List[Dict[str, Any]] = []
    for key_topic in sorted(topic_map.values(), key=lambda x: (x["domain"], x["topic"])):
        top_rules = list(key_topic.get("top_down_rules") or [])
        exp_rules = list(key_topic.get("experience_rules") or [])
        if rule_source == "experience-only":
            merged_rules = exp_rules
        elif rule_source == "knowledge-only":
            merged_rules = top_rules
        else:
            merged_rules = top_rules + exp_rules

        deduped: List[Dict[str, Any]] = []
        seen_ids = set()
        for rule in merged_rules:
            rid = str(rule.get("id") or "")
            if not rid or rid in seen_ids:
                continue
            deduped.append(rule)
            seen_ids.add(rid)

        target_domain = next((d for d in domains_out if d.get("name") == key_topic["domain"]), None)
        if target_domain is None:
            target_domain = {"name": key_topic["domain"], "topics": []}
            domains_out.append(target_domain)
        target_domain["topics"].append({"name": key_topic["topic"], "rules": deduped})

    total_topics = sum(len(d.get("topics") or []) for d in domains_out)
    total_rules = sum(len(t.get("rules") or []) for d in domains_out for t in (d.get("topics") or []))
    return {
        "metadata": {
            "version": "unified-2026-03",
            "generated_at": _dt.datetime.now().isoformat(),
            "rule_source": rule_source,
            "total_domains": len(domains_out),
            "total_topics": total_topics,
            "total_rules": total_rules,
            "top_down_rules_input": sum(len(t.get("top_down_rules") or []) for t in topic_map.values()),
            "experience_rules_input": sum(len(t.get("experience_rules") or []) for t in topic_map.values()),
            "experience_source_summary": distilled.get("summary", {}),
        },
        "domains": domains_out,
    }
