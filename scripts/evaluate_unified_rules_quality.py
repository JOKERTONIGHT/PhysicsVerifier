from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REMOVED_RUNTIME_FIELDS = {
    "includes",
    "excludes",
    "entry_cues",
    "related_topics",
    "related_clusters",
    "applicability",
    "negative_cues",
}


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return data


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _topic_key(domain: str, topic: str) -> str:
    return f"{domain}::{topic}"


def _ordered_unique(items: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        text = _text(item)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def _summary_bucket(summary: str) -> str:
    length = len(summary)
    if length == 0:
        return "missing"
    cjk_count = sum(1 for char in summary if "\u3400" <= char <= "\u9fff")
    min_length = 8 if cjk_count >= max(4, length // 2) else 12
    if length < min_length:
        return "too_short"
    if length > 420:
        return "too_long"
    return "ok"


def _issue(severity: str, code: str, message: str, evidence: Any = None) -> Dict[str, Any]:
    item = {"severity": severity, "code": code, "message": message}
    if evidence is not None:
        item["evidence"] = evidence
    return item


def _collect_catalog(catalog: Dict[str, Any]) -> Dict[str, Any]:
    domains = [item for item in catalog.get("domains", []) if isinstance(item, dict)]
    topics: List[Dict[str, Any]] = []
    clusters: List[Dict[str, Any]] = []
    rules: List[Dict[str, Any]] = []
    topic_rows: Dict[str, Dict[str, Any]] = {}
    removed_field_hits: List[Dict[str, str]] = []

    for domain in domains:
        domain_name = _text(domain.get("name"))
        for key in REMOVED_RUNTIME_FIELDS:
            if key in domain:
                removed_field_hits.append({"node": "domain", "id": _text(domain.get("id")), "field": key})
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = _text(topic.get("name"))
            key = _topic_key(domain_name, topic_name)
            topic_rules = [item for item in (topic.get("rules") or []) if isinstance(item, dict)]
            topic_clusters = [item for item in (topic.get("scenario_clusters") or []) if isinstance(item, dict)]
            topics.append(topic)
            for field in REMOVED_RUNTIME_FIELDS:
                if field in topic:
                    removed_field_hits.append({"node": "topic", "id": _text(topic.get("id")), "field": field})
            topic_rule_ids = {_text(rule.get("rule_id")) for rule in topic_rules if _text(rule.get("rule_id"))}
            cluster_rule_ids: set[str] = set()
            general_rule_ids: set[str] = set()
            for cluster in topic_clusters:
                clusters.append(cluster)
                cluster_id = _text(cluster.get("id") or cluster.get("cluster_id"))
                for field in REMOVED_RUNTIME_FIELDS:
                    if field in cluster:
                        removed_field_hits.append({"node": "cluster", "id": cluster_id, "field": field})
                ids = {_text(item) for item in (cluster.get("rule_ids") or []) if _text(item)}
                cluster_rule_ids.update(ids)
                if cluster_id == "general_reasoning":
                    general_rule_ids.update(ids)
            for rule in topic_rules:
                rules.append(rule)
                for field in REMOVED_RUNTIME_FIELDS:
                    if field in rule:
                        removed_field_hits.append({"node": "rule", "id": _text(rule.get("rule_id")), "field": field})
            topic_rows[key] = {
                "domain": domain_name,
                "topic": topic_name,
                "rule_count": len(topic_rules),
                "cluster_count": len(topic_clusters),
                "clustered_rule_count": len(cluster_rule_ids & topic_rule_ids),
                "unclustered_rule_count": len(topic_rule_ids - cluster_rule_ids),
                "general_reasoning_rule_count": len(general_rule_ids & topic_rule_ids),
                "rules": topic_rules,
                "clusters": topic_clusters,
            }
    return {
        "domains": domains,
        "topics": topics,
        "clusters": clusters,
        "rules": rules,
        "topic_rows": topic_rows,
        "removed_field_hits": removed_field_hits,
    }


def _summary_stats(items: List[Dict[str, Any]], *, label: str) -> Dict[str, Any]:
    buckets = Counter(_summary_bucket(_text(item.get("summary"))) for item in items)
    examples = [
        {
            "id": _text(item.get("id") or item.get("rule_id") or item.get("name")),
            "summary": _text(item.get("summary")),
        }
        for item in items
        if _summary_bucket(_text(item.get("summary"))) != "ok"
    ][:20]
    return {
        "node_type": label,
        "count": len(items),
        "missing": buckets["missing"],
        "too_short": buckets["too_short"],
        "too_long": buckets["too_long"],
        "ok": buckets["ok"],
        "problem_examples": examples,
    }


def _rule_completeness(rules: List[Dict[str, Any]]) -> Dict[str, Any]:
    required = ["rule_id", "title", "summary", "trigger", "check_logic", "error_type"]
    missing_by_field = {field: 0 for field in required}
    missing_symbolic_hint = 0
    empty_logic_examples: List[Dict[str, str]] = []
    for rule in rules:
        for field in required:
            if not _text(rule.get(field)):
                missing_by_field[field] += 1
        hint = rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {}
        if not hint:
            missing_symbolic_hint += 1
        if not _text(rule.get("trigger")) or not _text(rule.get("check_logic")):
            empty_logic_examples.append(
                {
                    "rule_id": _text(rule.get("rule_id")),
                    "title": _text(rule.get("title")),
                    "trigger": _text(rule.get("trigger")),
                    "check_logic": _text(rule.get("check_logic")),
                }
            )
    return {
        "rule_count": len(rules),
        "missing_by_field": missing_by_field,
        "missing_symbolic_hint": missing_symbolic_hint,
        "empty_logic_examples": empty_logic_examples[:20],
    }


def _cluster_stats(topic_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows = []
    total_rules = 0
    total_clustered = 0
    total_general = 0
    singleton_clusters = 0
    large_clusters = []
    for key, row in topic_rows.items():
        rule_count = int(row["rule_count"])
        clustered = int(row["clustered_rule_count"])
        general = int(row["general_reasoning_rule_count"])
        total_rules += rule_count
        total_clustered += clustered
        total_general += general
        for cluster in row["clusters"]:
            size = len(cluster.get("rule_ids") or [])
            if size == 1:
                singleton_clusters += 1
            if size >= 40:
                large_clusters.append(
                    {
                        "topic_key": key,
                        "cluster_id": _text(cluster.get("id") or cluster.get("cluster_id")),
                        "rule_count": size,
                    }
                )
        rows.append(
            {
                "topic_key": key,
                "rule_count": rule_count,
                "cluster_count": int(row["cluster_count"]),
                "clustered_rule_count": clustered,
                "unclustered_rule_count": int(row["unclustered_rule_count"]),
                "clustered_rule_ratio": round(clustered / rule_count, 4) if rule_count else 0.0,
                "general_reasoning_rule_ratio": round(general / rule_count, 4) if rule_count else 0.0,
            }
        )
    rows.sort(key=lambda item: (-item["rule_count"], item["topic_key"]))
    unclustered = [item for item in rows if item["rule_count"] > 0 and item["cluster_count"] == 0]
    low_coverage = [
        item for item in rows
        if item["rule_count"] >= 20 and item["clustered_rule_ratio"] < 0.5
    ]
    low_coverage.sort(key=lambda item: (item["clustered_rule_ratio"], -item["rule_count"], item["topic_key"]))
    return {
        "topic_count_with_rules": sum(1 for item in rows if item["rule_count"] > 0),
        "clustered_topic_count": sum(1 for item in rows if item["cluster_count"] > 0),
        "total_rule_count": total_rules,
        "clustered_rule_count": total_clustered,
        "unclustered_rule_count": total_rules - total_clustered,
        "clustered_rule_ratio": round(total_clustered / total_rules, 4) if total_rules else 0.0,
        "general_reasoning_rule_count": total_general,
        "general_reasoning_rule_ratio": round(total_general / total_rules, 4) if total_rules else 0.0,
        "unclustered_topic_count": len(unclustered),
        "unclustered_rule_count_from_topics": sum(int(item["rule_count"]) for item in unclustered),
        "top_unclustered_topics": unclustered[:20],
        "low_cluster_coverage_topics": low_coverage[:20],
        "singleton_cluster_count": singleton_clusters,
        "large_clusters": sorted(large_clusters, key=lambda item: (-item["rule_count"], item["topic_key"]))[:20],
        "topics": rows,
    }


def _duplication_stats(topic_rows: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    duplicate_title_groups: List[Dict[str, Any]] = []
    duplicate_summary_groups: List[Dict[str, Any]] = []
    exact_semantic_groups: List[Dict[str, Any]] = []
    for key, row in topic_rows.items():
        by_title: Dict[str, List[str]] = defaultdict(list)
        by_summary: Dict[str, List[str]] = defaultdict(list)
        by_semantic: Dict[str, List[str]] = defaultdict(list)
        for rule in row["rules"]:
            rule_id = _text(rule.get("rule_id"))
            title = _text(rule.get("title")).casefold()
            summary = _text(rule.get("summary")).casefold()
            semantic = "||".join([
                _text(rule.get("title")).casefold(),
                _text(rule.get("trigger")).casefold(),
                _text(rule.get("check_logic")).casefold(),
            ])
            if title:
                by_title[title].append(rule_id)
            if summary:
                by_summary[summary].append(rule_id)
            if semantic.strip("|"):
                by_semantic[semantic].append(rule_id)
        for title, ids in by_title.items():
            if len(ids) > 1:
                duplicate_title_groups.append({"topic_key": key, "title": title, "rule_ids": ids})
        for summary, ids in by_summary.items():
            if len(ids) > 1:
                duplicate_summary_groups.append({"topic_key": key, "summary": summary, "rule_ids": ids})
        for semantic, ids in by_semantic.items():
            if len(ids) > 1:
                exact_semantic_groups.append({"topic_key": key, "rule_ids": ids})
    return {
        "duplicate_title_group_count": len(duplicate_title_groups),
        "duplicate_summary_group_count": len(duplicate_summary_groups),
        "exact_semantic_duplicate_group_count": len(exact_semantic_groups),
        "duplicate_title_examples": duplicate_title_groups[:20],
        "duplicate_summary_examples": duplicate_summary_groups[:20],
        "exact_semantic_duplicate_examples": exact_semantic_groups[:20],
    }


def _proposal_stats(path: Path | None) -> Dict[str, Any]:
    if not path or not path.exists():
        return {"available": False}
    payload = _load_json(path)
    failures = [item for item in (payload.get("failures") or []) if isinstance(item, dict)]
    proposals = [item for item in (payload.get("proposals") or []) if isinstance(item, dict)]
    return {
        "available": True,
        "proposal_count": len(proposals),
        "failure_count": len(failures),
        "failures": failures,
        "metadata": payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {},
    }


def _runtime_eval_stats(path: Path | None, *, catalog_path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {"available": False}
    payload = _load_json(path)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    rows = [item for item in (payload.get("rows") or []) if isinstance(item, dict)]
    empty_rows = [row for row in rows if int(row.get("rule_count") or 0) == 0]
    high_rule_rows = [row for row in rows if int(row.get("rule_count") or 0) > 5]
    broad_topic_rows = [row for row in rows if int(row.get("topic_count") or 0) > 2]
    broad_cluster_rows = [row for row in rows if int(row.get("cluster_count") or 0) > 3]
    semantic_error_rows = [row for row in rows if _text(row.get("semantic_selection_error"))]
    stale = False
    if catalog_path.exists():
        stale = path.stat().st_mtime < catalog_path.stat().st_mtime
    return {
        "available": True,
        "stale": stale,
        "sample_count": int(summary.get("sample_count") or len(rows)),
        "semantic_error_count": int(summary.get("semantic_error_count") or len(semantic_error_rows)),
        "rule_selection_rate": float(summary.get("rule_selection_rate") or 0.0),
        "average_selected_rules": float(summary.get("average_selected_rules") or 0.0),
        "empty_rule_sample_ids": _ordered_unique(row.get("sample_id") for row in empty_rows),
        "high_rule_selection_sample_ids": _ordered_unique(row.get("sample_id") for row in high_rule_rows),
        "broad_topic_selection_sample_ids": _ordered_unique(row.get("sample_id") for row in broad_topic_rows),
        "broad_cluster_selection_sample_ids": _ordered_unique(row.get("sample_id") for row in broad_cluster_rows),
    }


def _score(report: Dict[str, Any]) -> Tuple[int, List[Dict[str, Any]]]:
    score = 100
    issues: List[Dict[str, Any]] = []
    schema = report["schema"]
    summaries = report["summary_quality"]
    rules = report["rule_completeness"]
    clusters = report["cluster_quality"]
    duplication = report["duplication"]
    proposals = report["cluster_proposals"]
    runtime = report["runtime_eval"]

    if schema["schema_profile"] != "semantic_navigation_tree_minimal":
        score -= 20
        issues.append(_issue("high", "schema_profile", "runtime catalog schema_profile is not semantic_navigation_tree_minimal", schema["schema_profile"]))
    if schema["removed_runtime_field_hit_count"]:
        score -= 15
        issues.append(_issue("high", "removed_runtime_fields", "removed runtime fields are present in catalog", schema["removed_runtime_field_examples"]))
    summary_problem_count = sum(
        item["missing"] + item["too_short"] + item["too_long"]
        for item in summaries.values()
    )
    if summary_problem_count:
        score -= min(20, summary_problem_count)
        issues.append(_issue("medium", "summary_quality", "some node summaries are missing or outside navigation length bounds", summary_problem_count))
    missing_rule_logic = rules["missing_by_field"]["trigger"] + rules["missing_by_field"]["check_logic"]
    if missing_rule_logic:
        score -= min(20, missing_rule_logic)
        issues.append(_issue("high", "rule_logic_missing", "some rules lack trigger or check_logic", rules["empty_logic_examples"]))
    if clusters["clustered_rule_ratio"] < 0.6:
        score -= 15
        issues.append(_issue("medium", "cluster_coverage", "less than 60% of rules are covered by scenario clusters", clusters["clustered_rule_ratio"]))
    if clusters["unclustered_topic_count"] > 30:
        score -= 10
        issues.append(_issue("medium", "unclustered_topics", "many topics with rules still have no scenario clusters", clusters["top_unclustered_topics"][:10]))
    if proposals.get("failure_count", 0):
        score -= min(5, int(proposals["failure_count"]))
        issues.append(_issue("low", "cluster_proposal_failures", "some cluster labeling topics failed and were not converted to generated blueprints", proposals["failures"]))
    if duplication["duplicate_summary_group_count"] > 100:
        score -= 5
        issues.append(_issue("low", "duplicate_summaries", "many rules share identical summaries; review for over-fine or near-duplicate rules", duplication["duplicate_summary_group_count"]))
    if runtime.get("available") and runtime.get("stale"):
        score -= 5
        issues.append(_issue("medium", "runtime_eval_stale", "runtime evaluation was produced before the current catalog and must be rerun", runtime))
    elif runtime.get("available"):
        if runtime["semantic_error_count"]:
            score -= 20
            issues.append(_issue("high", "runtime_semantic_errors", "runtime semantic tree selection still has model/parsing errors", runtime["semantic_error_count"]))
        if runtime["empty_rule_sample_ids"]:
            score -= min(10, len(runtime["empty_rule_sample_ids"]) * 3)
            issues.append(_issue("medium", "runtime_empty_rules", "some runtime samples reach the tree but select no rules", runtime["empty_rule_sample_ids"]))
        broad_samples = _ordered_unique(
            list(runtime["high_rule_selection_sample_ids"])
            + list(runtime["broad_topic_selection_sample_ids"])
            + list(runtime["broad_cluster_selection_sample_ids"])
        )
        if broad_samples:
            score -= min(8, len(broad_samples) * 2)
            issues.append(_issue("low", "runtime_overbroad_selection", "some runtime samples select too many topics, clusters, or rules", broad_samples))
    return max(0, score), issues


def evaluate_catalog_quality(
    *,
    catalog_path: Path,
    output_path: Path | None = None,
    cluster_proposals_path: Path | None = None,
    runtime_eval_path: Path | None = None,
) -> Dict[str, Any]:
    catalog = _load_json(catalog_path)
    collected = _collect_catalog(catalog)
    metadata = catalog.get("metadata") if isinstance(catalog.get("metadata"), dict) else {}
    summary_quality = {
        "domains": _summary_stats(collected["domains"], label="domain"),
        "topics": _summary_stats(collected["topics"], label="topic"),
        "clusters": _summary_stats(collected["clusters"], label="cluster"),
        "rules": _summary_stats(collected["rules"], label="rule"),
    }
    report = {
        "catalog": str(catalog_path),
        "schema": {
            "catalog_type": metadata.get("catalog_type"),
            "schema_profile": metadata.get("schema_profile"),
            "total_domains": metadata.get("total_domains"),
            "total_topics": metadata.get("total_topics"),
            "topics_with_rules": metadata.get("topics_with_rules"),
            "total_executable_rules": metadata.get("total_executable_rules"),
            "total_scenario_clusters": metadata.get("total_scenario_clusters"),
            "removed_runtime_field_hit_count": len(collected["removed_field_hits"]),
            "removed_runtime_field_examples": collected["removed_field_hits"][:20],
        },
        "summary_quality": summary_quality,
        "rule_completeness": _rule_completeness(collected["rules"]),
        "cluster_quality": _cluster_stats(collected["topic_rows"]),
        "duplication": _duplication_stats(collected["topic_rows"]),
        "cluster_proposals": _proposal_stats(cluster_proposals_path),
        "runtime_eval": _runtime_eval_stats(runtime_eval_path, catalog_path=catalog_path),
    }
    score, issues = _score(report)
    report["overall"] = {
        "quality_score": score,
        "status": "usable_with_known_gaps" if score >= 70 else "needs_rework",
        "issues": issues,
        "recommended_next_steps": [
            "补齐 cluster_proposals 中失败 topic，减少 missing generated clusters。",
            "优先处理低 cluster 覆盖且规则数高的 topic。",
            "抽样审查 duplicate summary/title 组，判断是否需要语义合并。",
            "用 top_down_verifier 做 30-100 条端到端检索命中率评估。",
            "优先复盘 runtime empty_rule_sample_ids 和 overbroad selection 样本。",
        ],
    }
    if output_path:
        _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate unified rules catalog quality for full pipeline readiness.")
    parser.add_argument("--catalog", default="catalogs/rules_unified_3000.json")
    parser.add_argument("--cluster-proposals", default="results/unified_rules_3000/cluster_proposals.json")
    parser.add_argument("--runtime-eval", default="results/unified_rules_3000/top_down_runtime_eval_30_fixed.json")
    parser.add_argument("--output", default="results/unified_rules_3000/rules_unified_quality_report.json")
    args = parser.parse_args()
    report = evaluate_catalog_quality(
        catalog_path=Path(args.catalog),
        cluster_proposals_path=Path(args.cluster_proposals) if args.cluster_proposals else None,
        runtime_eval_path=Path(args.runtime_eval) if args.runtime_eval else None,
        output_path=Path(args.output) if args.output else None,
    )
    compact = {
        "overall": report["overall"],
        "schema": report["schema"],
        "cluster_quality": {
            key: value
            for key, value in report["cluster_quality"].items()
            if key != "topics"
        },
        "duplication": {
            key: value
            for key, value in report["duplication"].items()
            if not key.endswith("_examples")
        },
        "runtime_eval": report["runtime_eval"],
    }
    print(json.dumps(compact, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
