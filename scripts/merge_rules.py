"""merge_rules.py – 将三个规则来源合并为统一规则库。

来源文件：
1. catalogs/rules_catalog_top_down.json   (知识规则: domain→topic→rules)
2. catalogs/rules_300_tagged.json         (经验规则: 平坦列表, 带 tags)
3. catalogs/semantic_experience_distilled_300.json  (提炼经验规则: 带 trigger/symbolic_hint)

输出：catalogs/rules_unified.json
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ----------------------------- Schema helpers -----------------------------

def _make_unified_rule(
    *,
    rule_id: str,
    title: str,
    description: str,
    check_logic: str = "",
    common_errors: Optional[List[str]] = None,
    source: str,
    source_file: str,
    error_type: Optional[str] = None,
    domain: str,
    topic: str,
    tag_type: Optional[str] = None,
    trigger: Optional[str] = None,
    symbolic_hint: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rule: Dict[str, Any] = {
        "id": rule_id,
        "title": title,
        "description": description,
        "source": source,
        "source_file": source_file,
        "tags": {
            "domain": domain,
            "topic": topic,
        },
    }
    if check_logic:
        rule["check_logic"] = check_logic
    if common_errors:
        rule["common_errors"] = common_errors
    if error_type:
        rule["error_type"] = error_type
    if tag_type:
        rule["tags"]["type"] = tag_type
    if trigger:
        rule["trigger"] = trigger
    if symbolic_hint and isinstance(symbolic_hint, dict):
        # Only include if it has a non-trivial primitive
        prim = str(symbolic_hint.get("primitive") or "none").strip()
        if prim not in ("", "none"):
            rule["symbolic_hint"] = symbolic_hint
    return rule


# ----------------------------- Source loaders -----------------------------

def load_knowledge_rules(path: Path) -> List[Dict[str, Any]]:
    """Load rules_catalog_top_down.json → flat list of unified rules."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rules: List[Dict[str, Any]] = []
    for domain in data.get("domains", []):
        domain_name = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []):
            topic_name = str(topic.get("name") or "Unknown")
            for r in topic.get("rules", []):
                rid = r.get("id")
                if not rid:
                    continue
                rules.append(_make_unified_rule(
                    rule_id=rid,
                    title=str(r.get("title") or ""),
                    description=str(r.get("description") or ""),
                    check_logic=str(r.get("check_logic") or ""),
                    common_errors=r.get("common_errors"),
                    source="knowledge",
                    source_file=path.name,
                    domain=domain_name,
                    topic=topic_name,
                ))
    return rules


def load_experience_tagged_rules(path: Path) -> List[Dict[str, Any]]:
    """Load rules_300_tagged.json → flat list of unified rules."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    rules: List[Dict[str, Any]] = []
    for r in data:
        rid = r.get("id")
        if not rid:
            continue
        tags = r.get("tags") if isinstance(r.get("tags"), dict) else {}
        error_type_raw = tags.get("type")  # "Check" etc., not error_type per se
        source_error_type = r.get("source_error_type")

        rules.append(_make_unified_rule(
            rule_id=rid,
            title=str(r.get("title") or ""),
            description=str(r.get("description") or ""),
            source="experience_tagged",
            source_file=path.name,
            error_type=source_error_type,
            domain=str(tags.get("domain") or "Unknown"),
            topic=str(tags.get("topic") or "Unknown"),
            tag_type=str(error_type_raw) if error_type_raw else None,
        ))
    return rules


def load_experience_distilled_rules(path: Path) -> List[Dict[str, Any]]:
    """Load semantic_experience_distilled_300.json → flat list of unified rules."""
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_rules = data.get("rules") if isinstance(data, dict) else []
    if not isinstance(raw_rules, list):
        return []

    rules: List[Dict[str, Any]] = []
    for r in raw_rules:
        if not isinstance(r, dict):
            continue
        rid = str(r.get("rule_id") or "")
        if not rid:
            continue

        # Normalize topic: distilled topics may look like "Electromagnetism / AC Circuits ..."
        raw_topic = str(r.get("topic") or "Unknown")
        # Strip domain prefix if duplicated in topic string
        domain = str(r.get("domain") or "Unknown")
        if "/" in raw_topic:
            # e.g. "Electromagnetism / AC Circuits (Impedance and Reactance)"
            parts = [p.strip() for p in raw_topic.split("/", 1)]
            if len(parts) == 2 and parts[0].lower() == domain.lower():
                raw_topic = parts[1]

        # Build description from check_logic (distilled rules don't have long descriptions)
        check_logic = str(r.get("check_logic") or "")
        title = str(r.get("title") or "")
        trigger = str(r.get("trigger") or "")
        desc_parts = []
        if trigger:
            desc_parts.append(f"触发条件：{trigger}")
        if check_logic:
            desc_parts.append(f"检查逻辑：{check_logic}")
        description = "\n".join(desc_parts) if desc_parts else check_logic

        rules.append(_make_unified_rule(
            rule_id=rid,
            title=title,
            description=description,
            check_logic=check_logic,
            source="experience",
            source_file=path.name,
            error_type=str(r.get("error_type") or ""),
            domain=domain,
            topic=raw_topic,
            trigger=trigger or None,
            symbolic_hint=r.get("symbolic_hint"),
        ))
    return rules


# ----------------------------- Merge logic -----------------------------

def _normalize_key(domain: str, topic: str) -> Tuple[str, str]:
    return domain.strip(), topic.strip()


def merge_into_hierarchy(all_rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group flat rules list into domain → topic hierarchy, preserving order."""
    # domain_name -> topic_name -> [rules]
    tree: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    seen_ids: Set[str] = set()

    for rule in all_rules:
        rid = rule.get("id")
        if rid in seen_ids:
            continue
        seen_ids.add(rid)

        domain = rule.get("tags", {}).get("domain", "Unknown")
        topic = rule.get("tags", {}).get("topic", "Unknown")
        domain, topic = _normalize_key(domain, topic)

        tree.setdefault(domain, {}).setdefault(topic, []).append(rule)

    # Convert to output format
    domains_out: List[Dict[str, Any]] = []
    for domain_name in sorted(tree.keys()):
        topics_out: List[Dict[str, Any]] = []
        for topic_name in sorted(tree[domain_name].keys()):
            rules = tree[domain_name][topic_name]
            topics_out.append({
                "name": topic_name,
                "rules": rules,
            })
        domains_out.append({
            "name": domain_name,
            "topics": topics_out,
        })

    return {
        "metadata": {
            "version": "1.0",
            "generated_at": datetime.datetime.now().isoformat(),
            "total_rules": len(seen_ids),
            "total_domains": len(domains_out),
            "total_topics": sum(len(d["topics"]) for d in domains_out),
        },
        "domains": domains_out,
    }


# ----------------------------- CLI entry point -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Merge knowledge and experience rules into a unified catalog.")
    parser.add_argument(
        "--knowledge",
        type=str,
        default="catalogs/rules_catalog_top_down.json",
        help="Path to knowledge rules catalog (top-down).",
    )
    parser.add_argument(
        "--experience-tagged",
        type=str,
        default="catalogs/rules_300_tagged.json",
        help="Path to tagged experience rules.",
    )
    parser.add_argument(
        "--experience-distilled",
        type=str,
        default="catalogs/semantic_experience_distilled_300.json",
        help="Path to distilled experience rules.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="catalogs/rules_unified.json",
        help="Output path for the unified rules catalog.",
    )

    args = parser.parse_args()

    knowledge_path = Path(args.knowledge)
    tagged_path = Path(args.experience_tagged)
    distilled_path = Path(args.experience_distilled)
    output_path = Path(args.output)

    all_rules: List[Dict[str, Any]] = []

    # 1. Knowledge rules (load first – they form the primary hierarchy)
    if knowledge_path.exists():
        knowledge_rules = load_knowledge_rules(knowledge_path)
        print(f"  Knowledge rules loaded: {len(knowledge_rules)}")
        all_rules.extend(knowledge_rules)
    else:
        print(f"  Warning: Knowledge rules not found at {knowledge_path}")

    # 2. Tagged experience rules
    if tagged_path.exists():
        tagged_rules = load_experience_tagged_rules(tagged_path)
        print(f"  Tagged experience rules loaded: {len(tagged_rules)}")
        all_rules.extend(tagged_rules)
    else:
        print(f"  Warning: Tagged experience rules not found at {tagged_path}")

    # 3. Distilled experience rules
    if distilled_path.exists():
        distilled_rules = load_experience_distilled_rules(distilled_path)
        print(f"  Distilled experience rules loaded: {len(distilled_rules)}")
        all_rules.extend(distilled_rules)
    else:
        print(f"  Warning: Distilled experience rules not found at {distilled_path}")

    # Merge
    unified = merge_into_hierarchy(all_rules)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(unified, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    meta = unified["metadata"]
    print(f"\nDone. Unified catalog written to: {output_path}")
    print(f"  Total rules:   {meta['total_rules']}")
    print(f"  Total domains: {meta['total_domains']}")
    print(f"  Total topics:  {meta['total_topics']}")


if __name__ == "__main__":
    main()
