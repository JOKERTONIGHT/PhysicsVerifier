from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _has_items(value: Any) -> bool:
    return isinstance(value, list) and any(str(item or "").strip() for item in value)


def _iter_rules(catalog: Dict[str, Any]) -> Iterable[tuple[str, str, Dict[str, Any]]]:
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = str(topic.get("name") or "Unknown")
            for rule in topic.get("rules", []) or []:
                if isinstance(rule, dict):
                    yield domain_name, topic_name, rule


def _iter_topics(catalog: Dict[str, Any]) -> Iterable[tuple[str, str, Dict[str, Any]]]:
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            if isinstance(topic, dict):
                yield domain_name, str(topic.get("name") or "Unknown"), topic


def _rule_id(rule: Dict[str, Any]) -> str:
    return str(rule.get("rule_id") or rule.get("id") or "").strip()


def _load_fp_rule_counts(path: str) -> Counter[str]:
    if not path:
        return Counter()
    data = _load_json(path)
    rows: List[Dict[str, Any]] = []
    if isinstance(data, dict):
        replay = data.get("false_positive_replay")
        if isinstance(replay, list):
            rows = [item for item in replay if isinstance(item, dict)]
        elif isinstance(data.get("details"), list):
            rows = [item for item in data["details"] if isinstance(item, dict)]
    elif isinstance(data, list):
        rows = [item for item in data if isinstance(item, dict)]

    counts: Counter[str] = Counter()
    for row in rows:
        rid = str(row.get("rule_id") or row.get("rule") or "").strip()
        if not rid and isinstance(row.get("rule_match"), dict):
            rid = str(row["rule_match"].get("rule_id") or "").strip()
        if rid:
            counts[rid] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit runtime metadata coverage for a unified rule catalog.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--fp-metrics", default="", help="Optional error_metrics.json containing false_positive_replay.")
    parser.add_argument("--output", default="")
    parser.add_argument(
        "--rule-ids-output",
        default="",
        help="Optional path to write FP rule ids as a JSON list, ordered by FP count.",
    )
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    catalog = _load_json(args.catalog)
    fp_counts = _load_fp_rule_counts(args.fp_metrics)

    required_fields = [
        "match_features",
        "support",
        "preconditions",
        "violation_signatures",
        "negative_conditions",
        "evidence_requirements",
        "symbolic_hint",
    ]
    optional_fields = ["llm_hints", "source_rule_ids"]
    fields = required_fields + optional_fields
    coverage = {field: 0 for field in fields}
    total = 0
    missing_rows: List[Dict[str, Any]] = []
    missing_optional_rows: List[Dict[str, Any]] = []
    by_topic: Counter[str] = Counter()
    fp_rows: List[Dict[str, Any]] = []
    catalog_rule_ids: set[str] = set()

    for domain, topic, rule in _iter_rules(catalog):
        total += 1
        rid = _rule_id(rule)
        if rid:
            catalog_rule_ids.add(rid)
        topic_key = f"{domain}::{topic}"
        by_topic[topic_key] += 1
        for field in fields:
            value = rule.get(field)
            if field in {"match_features", "support", "llm_hints", "symbolic_hint"}:
                ok = isinstance(value, dict) and bool(value)
            else:
                ok = _has_items(value)
            if ok:
                coverage[field] += 1
        missing = [
            field
            for field in required_fields
            if not (
                (isinstance(rule.get(field), dict) and bool(rule.get(field)))
                if field in {"match_features", "support", "llm_hints", "symbolic_hint"}
                else _has_items(rule.get(field))
            )
        ]
        if missing:
            missing_rows.append(
                {
                    "rule_id": rid,
                    "domain": domain,
                    "topic": topic,
                    "title": str(rule.get("title") or ""),
                    "missing": missing,
                    "fp_count": int(fp_counts.get(rid, 0)),
                }
            )
        missing_optional = [
            field
            for field in optional_fields
            if not (
                isinstance(rule.get(field), dict) and bool(rule.get(field))
                if field == "llm_hints"
                else _has_items(rule.get(field))
            )
        ]
        if missing_optional:
            missing_optional_rows.append(
                {
                    "rule_id": rid,
                    "domain": domain,
                    "topic": topic,
                    "title": str(rule.get("title") or ""),
                    "missing": missing_optional,
                    "fp_count": int(fp_counts.get(rid, 0)),
                }
            )
        if fp_counts.get(rid, 0):
            fp_rows.append(
                {
                    "rule_id": rid,
                    "fp_count": int(fp_counts[rid]),
                    "domain": domain,
                    "topic": topic,
                    "title": str(rule.get("title") or ""),
                    "missing": missing,
                }
            )

    fp_rows.sort(key=lambda item: (-int(item["fp_count"]), item["domain"], item["topic"], item["rule_id"]))
    missing_rows.sort(key=lambda item: (-int(item["fp_count"]), item["domain"], item["topic"], item["rule_id"]))
    missing_optional_rows.sort(
        key=lambda item: (-int(item["fp_count"]), item["domain"], item["topic"], item["rule_id"])
    )

    topic_fields = ["retrieval_hints", "knowledge_reference", "tagged_reference"]
    topic_coverage = {field: 0 for field in topic_fields}
    topic_total = 0
    missing_topic_rows: List[Dict[str, Any]] = []
    for domain, topic, topic_obj in _iter_topics(catalog):
        topic_total += 1
        missing_topic = []
        for field in topic_fields:
            ok = isinstance(topic_obj.get(field), dict) and bool(topic_obj.get(field))
            if ok:
                topic_coverage[field] += 1
            else:
                missing_topic.append(field)
        if missing_topic:
            missing_topic_rows.append({"domain": domain, "topic": topic, "missing": missing_topic})

    unmatched_fp_rule_ids = [rid for rid, _ in fp_counts.most_common() if rid not in catalog_rule_ids]
    required_rule_coverage_complete = all(coverage[field] == total for field in required_fields)
    topic_routing_coverage_complete = topic_coverage["retrieval_hints"] == topic_total

    output = {
        "catalog": args.catalog,
        "total_rules": total,
        "coverage": coverage,
        "coverage_ratio": {field: (coverage[field] / total if total else 0.0) for field in fields},
        "required_rule_fields": required_fields,
        "optional_rule_fields": optional_fields,
        "topic_total": topic_total,
        "topic_coverage": topic_coverage,
        "topic_coverage_ratio": {
            field: (topic_coverage[field] / topic_total if topic_total else 0.0) for field in topic_fields
        },
        "runtime_readiness": {
            "required_rule_coverage_complete": required_rule_coverage_complete,
            "topic_routing_coverage_complete": topic_routing_coverage_complete,
            "ready": required_rule_coverage_complete and topic_routing_coverage_complete,
        },
        "topic_count": len(by_topic),
        "largest_topics": [
            {"topic": topic, "rule_count": count}
            for topic, count in by_topic.most_common(max(0, int(args.top)))
        ],
        "fp_rules": fp_rows[: max(0, int(args.top))],
        "fp_rule_ids": [row["rule_id"] for row in fp_rows[: max(0, int(args.top))]],
        "unmatched_fp_rule_ids": unmatched_fp_rule_ids[: max(0, int(args.top))],
        "missing_runtime_metadata": missing_rows[: max(0, int(args.top))],
        "missing_optional_metadata": missing_optional_rows[: max(0, int(args.top))],
        "missing_topic_metadata": missing_topic_rows[: max(0, int(args.top))],
    }

    text = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.rule_ids_output:
        out_ids = Path(args.rule_ids_output)
        out_ids.parent.mkdir(parents=True, exist_ok=True)
        out_ids.write_text(json.dumps(output["fp_rule_ids"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Keep the JSON artifact human-readable while avoiding mojibake in Windows
    # consoles whose active code page cannot render all catalog titles.
    print(json.dumps(output, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
