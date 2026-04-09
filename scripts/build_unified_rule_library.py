from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _normalize_topic(domain_name: str, topic_name: str) -> str:
    t = str(topic_name or "Unknown").strip()
    d = str(domain_name or "Unknown").strip()
    if "/" in t:
        parts = [p.strip() for p in t.split("/", 1)]
        if len(parts) == 2 and parts[0].lower() == d.lower():
            return parts[1]
    return t


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


def build_unified_library(
    rules_catalog: Dict[str, Any],
    distilled: Dict[str, Any],
    *,
    rule_source: str = "hybrid",
) -> Dict[str, Any]:
    topic_map: Dict[str, Dict[str, Any]] = {}

    for domain in rules_catalog.get("domains", []) or []:
        domain_name = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            topic_name = _normalize_topic(domain_name, str(topic.get("name") or "Unknown"))
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
        topic_name = _normalize_topic(domain_name, str(er.get("topic") or "Unknown"))
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
        bucket["experience_rules"].append(_experience_rule_to_catalog_rule(er))

    topics = list(topic_map.values())
    topics.sort(key=lambda x: (x["domain"], x["topic"]))

    domains_out: List[Dict[str, Any]] = []
    for key_topic in topics:
        domain_name = key_topic["domain"]
        topic_name = key_topic["topic"]
        top_rules = list(key_topic.get("top_down_rules") or [])
        exp_rules = list(key_topic.get("experience_rules") or [])

        if rule_source == "experience-only":
            merged_rules = exp_rules
        elif rule_source == "knowledge-only":
            merged_rules = top_rules
        else:
            merged_rules = top_rules + exp_rules

        # Stable dedupe by rule id
        deduped: List[Dict[str, Any]] = []
        seen_ids = set()
        for r in merged_rules:
            rid = str(r.get("id") or "")
            if not rid or rid in seen_ids:
                continue
            deduped.append(r)
            seen_ids.add(rid)

        target_domain = None
        for d in domains_out:
            if d.get("name") == domain_name:
                target_domain = d
                break
        if target_domain is None:
            target_domain = {"name": domain_name, "topics": []}
            domains_out.append(target_domain)

        target_domain["topics"].append({"name": topic_name, "rules": deduped})

    total_topics = sum(len(d.get("topics") or []) for d in domains_out)
    total_rules = sum(len(t.get("rules") or []) for d in domains_out for t in (d.get("topics") or []))
    total_exp_rules = sum(len(t.get("experience_rules") or []) for t in topics)
    total_top_rules = sum(len(t.get("top_down_rules") or []) for t in topics)

    return {
        "metadata": {
            "version": "unified-2026-03",
            "generated_at": datetime.datetime.now().isoformat(),
            "rule_source": rule_source,
            "total_domains": len(domains_out),
            "total_topics": total_topics,
            "total_rules": total_rules,
            "top_down_rules_input": total_top_rules,
            "experience_rules_input": total_exp_rules,
            "experience_source_summary": distilled.get("summary", {}),
        },
        "domains": domains_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified rule library from top-down catalog and distilled experience rules.")
    parser.add_argument("--rules-catalog", type=str, default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--experience-distilled", type=str, required=True)
    parser.add_argument("--output", type=str, default="catalogs/unified_rule_library.json")
    parser.add_argument(
        "--rule-source",
        type=str,
        default="experience-only",
        choices=["experience-only", "hybrid", "knowledge-only"],
        help="Which rule set to emit into final verifier-compatible catalog.",
    )
    args = parser.parse_args()

    catalog = _load_json(Path(args.rules_catalog))
    distilled = _load_json(Path(args.experience_distilled))
    unified = build_unified_library(catalog, distilled, rule_source=args.rule_source)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(unified, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Unified rule library saved to {out_path}")
    meta = unified.get("metadata", {}) if isinstance(unified, dict) else {}
    print(
        "Summary:",
        json.dumps(
            {
                "rule_source": meta.get("rule_source"),
                "total_domains": meta.get("total_domains"),
                "total_topics": meta.get("total_topics"),
                "total_rules": meta.get("total_rules"),
            },
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
