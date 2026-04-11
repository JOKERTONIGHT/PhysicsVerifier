"""Build unified_rules v2 from knowledge skeleton + distilled experience rules.

Inputs:
1. catalogs/rules_catalog_top_down.json
2. catalogs/rules_300_tagged.json
3. catalogs/semantic_experience_distilled_300.json

Output:
- catalogs/rules_unified.json

The v2 catalog keeps the outer domain/topic/rules skeleton so downstream code can
still inspect the catalog shape, but the internal semantics are different:
- topic.rules contains only distilled experience rule leaves
- knowledge rules are moved to topic.knowledge_reference
- tagged experience rules are moved to topic.tagged_reference
- retrieval_hints and clusters are added for offline matching analysis
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.unified_retrieval import (
    build_scene_keywords,
    build_topic_required_symbols,
    classify_rule_scope,
    extract_keywords,
    norm_text,
    ordered_unique,
    normalize_rule_for_retrieval,
    refine_topic_hints,
)

CLUSTER_TOPIC_THRESHOLD = 12
CLUSTER_BUCKET_THRESHOLD = 3


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm_text(text: Any) -> str:
    return norm_text(text)


def _norm_key(value: Any) -> str:
    return _norm_text(value).lower()


def _ordered_unique(values: Iterable[str]) -> List[str]:
    return ordered_unique(values)


def _extract_keywords(texts: Iterable[str], *, max_keywords: int) -> List[str]:
    return extract_keywords(texts, max_keywords=max_keywords)


def _normalize_topic(domain: str, topic: str) -> str:
    norm_domain = _norm_text(domain)
    norm_topic = _norm_text(topic)
    if "/" in norm_topic:
        left, right = [part.strip() for part in norm_topic.split("/", 1)]
        if left.casefold() == norm_domain.casefold():
            norm_topic = right
    return norm_topic


def _topic_key(domain: str, topic: str) -> str:
    return f"{_norm_key(domain)}::{_norm_key(topic)}"


def _safe_symbolic_hint(raw_hint: Any) -> Dict[str, Any]:
    hint = raw_hint if isinstance(raw_hint, dict) else {}
    primitive = _norm_text(hint.get("primitive") or "none") or "none"
    canonical = _norm_text(hint.get("canonical") or "")
    required_symbols = _ordered_unique(str(item) for item in (hint.get("required_symbols") or []))
    return {
        "primitive": primitive,
        "canonical": canonical,
        "required_symbols": required_symbols,
    }


def _build_match_features(title: str, trigger: str, check_logic: str, symbolic_hint: Dict[str, Any]) -> Dict[str, Any]:
    required_symbols = _ordered_unique(str(item) for item in symbolic_hint.get("required_symbols", []))
    primitive = _norm_text(symbolic_hint.get("primitive") or "none") or "none"
    trigger_keywords = _extract_keywords([title, trigger], max_keywords=8)
    object_keywords = _extract_keywords([check_logic], max_keywords=8)
    return {
        "trigger_keywords": trigger_keywords,
        "object_keywords": object_keywords,
        "required_symbols": required_symbols,
        "primitive": primitive,
    }


def _cluster_keywords(rules: List[Dict[str, Any]], *, max_keywords: int = 8) -> List[str]:
    counter: Counter[str] = Counter()
    display: Dict[str, str] = {}
    order: Dict[str, int] = {}

    for rule in rules:
        features = rule.get("match_features") if isinstance(rule.get("match_features"), dict) else {}
        tokens = list(features.get("trigger_keywords") or []) + list(features.get("object_keywords") or [])
        for token in tokens:
            item = _norm_text(token)
            if not item:
                continue
            key = item.casefold()
            counter[key] += 1
            display.setdefault(key, item)
            order.setdefault(key, len(order))

    ranked = sorted(counter.items(), key=lambda item: (-item[1], order[item[0]], item[0]))
    return [display[key] for key, _ in ranked[:max_keywords]]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", _norm_text(value).lower()).strip("_")
    return slug or "unknown"


def _build_clusters(rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(rules) < CLUSTER_TOPIC_THRESHOLD:
        return []

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for rule in rules:
        error_type = _norm_text(rule.get("error_type") or "unknown") or "unknown"
        buckets.setdefault(error_type, []).append(rule)

    clusters: List[Dict[str, Any]] = []
    for error_type, members in sorted(buckets.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(members) < CLUSTER_BUCKET_THRESHOLD:
            continue
        clusters.append(
            {
                "cluster_id": f"cluster_{_slugify(error_type)}",
                "label": error_type,
                "error_type": error_type,
                "keywords": _cluster_keywords(members),
                "rule_ids": [str(rule["rule_id"]) for rule in members],
            }
        )
    return clusters


def _build_topic_skeleton(knowledge_data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    domains_out: List[Dict[str, Any]] = []
    states: Dict[str, Dict[str, Any]] = {}

    for domain in knowledge_data.get("domains", []) or []:
        domain_name = _norm_text(domain.get("name") or "Unknown")
        domain_out = {"name": domain_name, "topics": []}

        for topic in domain.get("topics", []) or []:
            topic_name = _norm_text(topic.get("name") or "Unknown")
            knowledge_rules = [item for item in (topic.get("rules") or []) if isinstance(item, dict)]
            knowledge_rule_ids = _ordered_unique(str(item.get("id") or "") for item in knowledge_rules)
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
                "knowledge_reference": {
                    "rule_ids": knowledge_rule_ids,
                    "keywords": _extract_keywords(knowledge_texts, max_keywords=16),
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

            key = _topic_key(domain_name, topic_name)
            states[key] = {
                "domain": domain_name,
                "topic": topic_name,
                "entry": entry,
                "tagged_keyword_texts": [],
            }

        domains_out.append(domain_out)

    return domains_out, states


def _attach_distilled_rules(states: Dict[str, Dict[str, Any]], distilled_data: Dict[str, Any]) -> None:
    raw_rules = distilled_data.get("rules") if isinstance(distilled_data, dict) else []
    if not isinstance(raw_rules, list):
        raise ValueError("Distilled experience data must contain a top-level 'rules' list.")

    unmatched: List[str] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, dict):
            continue

        domain = _norm_text(raw_rule.get("domain") or "Unknown")
        topic = _normalize_topic(domain, str(raw_rule.get("topic") or "Unknown"))
        key = _topic_key(domain, topic)
        state = states.get(key)
        if state is None:
            unmatched.append(f"{domain} / {topic} :: {raw_rule.get('rule_id')}")
            continue

        symbolic_hint = _safe_symbolic_hint(raw_rule.get("symbolic_hint"))
        title = _norm_text(raw_rule.get("title") or "")
        trigger = _norm_text(raw_rule.get("trigger") or "")
        check_logic = _norm_text(raw_rule.get("check_logic") or "")
        rule_leaf = {
            "rule_id": _norm_text(raw_rule.get("rule_id") or ""),
            "title": title,
            "trigger": trigger,
            "check_logic": check_logic,
            "error_type": _norm_text(raw_rule.get("error_type") or "logic") or "logic",
            "scope": classify_rule_scope(
                title=title,
                trigger=trigger,
                check_logic=check_logic,
            ),
            "symbolic_hint": symbolic_hint,
            "support": {
                "count": int(raw_rule.get("count") or 0),
                "sample_ids": _ordered_unique(str(item) for item in (raw_rule.get("sample_ids") or [])),
            },
            "match_features": _build_match_features(title, trigger, check_logic, symbolic_hint),
        }
        rule_leaf = normalize_rule_for_retrieval(rule_leaf)
        state["entry"]["rules"].append(rule_leaf)

    if unmatched:
        uniq = _ordered_unique(unmatched)
        preview = "\n".join(f"- {item}" for item in uniq[:10])
        raise ValueError(
            "Distilled rules contain topics that do not map to the knowledge skeleton.\n"
            f"Unmatched unique topics/rules: {len(uniq)}\n{preview}"
        )


def _attach_tagged_reference(states: Dict[str, Dict[str, Any]], tagged_data: Any) -> int:
    if not isinstance(tagged_data, list):
        raise ValueError("Tagged experience data must be a JSON list.")

    mapped_rules = 0
    for raw_rule in tagged_data:
        if not isinstance(raw_rule, dict):
            continue
        tags = raw_rule.get("tags") if isinstance(raw_rule.get("tags"), dict) else {}
        domain = _norm_text(tags.get("domain") or "Unknown")
        topic = _normalize_topic(domain, str(tags.get("topic") or "Unknown"))
        key = _topic_key(domain, topic)
        state = states.get(key)
        if state is None:
            continue

        title = _norm_text(raw_rule.get("title") or "")
        description = _norm_text(raw_rule.get("description") or "")
        source_id = _norm_text(raw_rule.get("id") or "")

        ref = state["entry"]["tagged_reference"]
        ref["source_ids"].append(source_id)
        ref["titles"].append(title)
        ref["aliases"].append(title)
        state["tagged_keyword_texts"].extend([title, description])
        mapped_rules += 1

    for state in states.values():
        ref = state["entry"]["tagged_reference"]
        ref["source_ids"] = _ordered_unique(ref["source_ids"])
        ref["titles"] = _ordered_unique(ref["titles"])
        ref["aliases"] = _ordered_unique(ref["aliases"])
        ref["keywords"] = _extract_keywords(state["tagged_keyword_texts"], max_keywords=16)

    return mapped_rules


def _finalize_topics(states: Dict[str, Dict[str, Any]]) -> None:
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
            if _norm_text(rule.get("scope") or "domain") != "domain":
                continue
            domain_rule_texts.extend(
                [
                    _norm_text(rule.get("title") or ""),
                    _norm_text(rule.get("trigger") or ""),
                ]
            )

        topic_keywords = _ordered_unique(
            list(_extract_keywords([state["topic"]], max_keywords=8))
            + list(entry["knowledge_reference"].get("keywords") or [])
            + list(entry["tagged_reference"].get("keywords") or [])
        )[:20]
        scene_keywords = build_scene_keywords(
            topic_name=state["topic"],
            tagged_titles=entry["tagged_reference"].get("titles") or [],
            tagged_aliases=entry["tagged_reference"].get("aliases") or [],
            rule_texts=domain_rule_texts,
        )
        scene_keywords, topic_keywords = refine_topic_hints(
            scene_keywords=scene_keywords,
            topic_keywords=topic_keywords,
            rule_texts=domain_rule_texts,
        )

        entry["retrieval_hints"] = {
            "scene_keywords": scene_keywords,
            "topic_keywords": topic_keywords,
            "required_symbols": build_topic_required_symbols(rules),
        }
        entry["clusters"] = _build_clusters(rules)


def build_unified_catalog_from_data(
    knowledge_data: Dict[str, Any],
    distilled_data: Dict[str, Any],
    tagged_data: Any,
) -> Dict[str, Any]:
    domains_out, states = _build_topic_skeleton(knowledge_data)
    _attach_distilled_rules(states, distilled_data)
    mapped_tagged_rules = _attach_tagged_reference(states, tagged_data)
    _finalize_topics(states)

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
        },
        "domains": domains_out,
    }


def build_unified_catalog(knowledge_path: Path, distilled_path: Path, tagged_path: Path) -> Dict[str, Any]:
    knowledge_data = _load_json(knowledge_path)
    distilled_data = _load_json(distilled_path)
    tagged_data = _load_json(tagged_path)
    return build_unified_catalog_from_data(knowledge_data, distilled_data, tagged_data)


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
    _write_json(output_path, catalog)

    meta = catalog["metadata"]
    print(f"Done. Unified v2 catalog written to: {output_path}")
    print(f"  Domains:             {meta['total_domains']}")
    print(f"  Topics:              {meta['total_topics']}")
    print(f"  Topics with rules:   {meta['topics_with_rules']}")
    print(f"  Executable rules:    {meta['total_executable_rules']}")
    print(f"  Tagged refs mapped:  {meta['mapped_tagged_reference_rules']}")
    print(f"  Clustered topics:    {meta['clustered_topics']}")
    print(f"  Total clusters:      {meta['total_clusters']}")


if __name__ == "__main__":
    main()