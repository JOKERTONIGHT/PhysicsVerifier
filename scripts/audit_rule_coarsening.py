from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _rule_ids(payload: Dict[str, Any], *, source: str) -> set[str]:
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError(f"{source} must contain a rules array.")
    ids = [str(rule.get("rule_id") or "").strip() for rule in rules if isinstance(rule, dict)]
    if any(not rule_id for rule_id in ids):
        raise ValueError(f"{source} contains an empty rule_id.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{source} contains duplicate rule_ids.")
    return set(ids)


def _catalog_records(catalog: Dict[str, Any]) -> tuple[set[str], Counter[str], List[int]]:
    rule_ids: set[str] = set()
    assignments: Counter[str] = Counter()
    residual_sizes: List[int] = []
    for domain in catalog.get("domains", []) or []:
        for topic in domain.get("topics", []) or []:
            for rule in topic.get("rules", []) or []:
                rule_id = str(rule.get("rule_id") or "").strip()
                if rule_id in rule_ids:
                    raise ValueError(f"Catalog contains duplicate rule_id: {rule_id}")
                rule_ids.add(rule_id)
            for cluster in topic.get("scenario_clusters", []) or []:
                cluster_id = str(cluster.get("id") or cluster.get("cluster_id") or "").strip()
                cluster_rule_ids = [
                    str(rule_id).strip()
                    for rule_id in (cluster.get("rule_ids") or [])
                    if str(rule_id).strip()
                ]
                assignments.update(cluster_rule_ids)
                if cluster_id.startswith("residual_rules_"):
                    residual_sizes.append(len(cluster_rule_ids))
    return rule_ids, assignments, residual_sizes


def audit_rule_coarsening(
    *,
    candidates: Dict[str, Any],
    generalized: Dict[str, Any],
    formal: Dict[str, Any],
    catalog: Dict[str, Any],
    residual_bucket_limit: int = 12,
) -> Dict[str, Any]:
    candidate_ids = _rule_ids(candidates, source="candidate bank")
    generated_ids = _rule_ids(generalized, source="generalized rules")
    formal_ids = _rule_ids(formal, source="formal rules")
    catalog_ids, assignments, residual_sizes = _catalog_records(catalog)

    generated_rules = {
        str(rule.get("rule_id") or "").strip(): rule
        for rule in generalized.get("rules", []) or []
        if isinstance(rule, dict)
    }
    insufficient_support = sorted(
        rule_id
        for rule_id, rule in generated_rules.items()
        if len({str(item) for item in (rule.get("sample_ids") or []) if str(item)}) < 2
        or int(rule.get("count") or 0) < 2
    )
    mapped_candidate_ids = {
        str(candidate_id)
        for result in generalized.get("cluster_results", []) or []
        if isinstance(result, dict)
        for mapping in result.get("mappings", []) or []
        if isinstance(mapping, dict)
        for candidate_id in mapping.get("source_candidate_ids", []) or []
        if str(candidate_id)
    }
    pending_candidate_ids = {
        str(candidate_id)
        for candidate_id in generalized.get("pending_candidate_ids", []) or []
        if str(candidate_id)
    }
    accounted_candidate_ids = mapped_candidate_ids | pending_candidate_ids
    assignment_errors = sorted(
        rule_id
        for rule_id in catalog_ids
        if assignments.get(rule_id, 0) != 1
    )
    unknown_cluster_rule_ids = sorted(set(assignments) - catalog_ids)
    general_reasoning_cluster_count = sum(
        1
        for domain in catalog.get("domains", []) or []
        for topic in domain.get("topics", []) or []
        for cluster in topic.get("scenario_clusters", []) or []
        if str(cluster.get("id") or cluster.get("cluster_id") or "") == "general_reasoning"
    )

    gates = {
        "candidate_accounting_complete": accounted_candidate_ids == candidate_ids,
        "mapped_and_pending_disjoint": not (mapped_candidate_ids & pending_candidate_ids),
        "generated_rules_have_multi_sample_support": not insufficient_support,
        "generated_rules_enter_formal_bank": generated_ids <= formal_ids,
        "formal_catalog_rule_conservation": formal_ids == catalog_ids,
        "each_catalog_rule_reachable_once": not assignment_errors,
        "cluster_references_known_rules": not unknown_cluster_rule_ids,
        "residual_bucket_size_bounded": all(
            size <= residual_bucket_limit for size in residual_sizes
        ),
        "no_general_reasoning_catchall": general_reasoning_cluster_count == 0,
        "candidate_rules_reduced_by_generalization": len(generated_ids) < len(candidate_ids),
    }
    return {
        "complete": all(gates.values()),
        "gates": gates,
        "counts": {
            "candidate_rules": len(candidate_ids),
            "mapped_candidates": len(mapped_candidate_ids),
            "pending_candidates": len(pending_candidate_ids),
            "generated_multi_sample_rules": len(generated_ids),
            "preserved_baseline_rules": len(formal_ids - generated_ids),
            "formal_rules": len(formal_ids),
            "catalog_rules": len(catalog_ids),
            "scenario_clusters": sum(
                1
                for domain in catalog.get("domains", []) or []
                for topic in domain.get("topics", []) or []
                for _cluster in topic.get("scenario_clusters", []) or []
            ),
            "residual_buckets": len(residual_sizes),
            "largest_residual_bucket": max(residual_sizes, default=0),
            "general_reasoning_clusters": general_reasoning_cluster_count,
        },
        "errors": {
            "unaccounted_candidate_ids": sorted(candidate_ids - accounted_candidate_ids),
            "unknown_accounted_candidate_ids": sorted(accounted_candidate_ids - candidate_ids),
            "mapped_pending_overlap_ids": sorted(
                mapped_candidate_ids & pending_candidate_ids
            ),
            "insufficient_support_generated_rule_ids": insufficient_support,
            "generated_rules_missing_from_formal": sorted(generated_ids - formal_ids),
            "formal_rules_missing_from_catalog": sorted(formal_ids - catalog_ids),
            "unexpected_catalog_rule_ids": sorted(catalog_ids - formal_ids),
            "catalog_assignment_errors": assignment_errors,
            "unknown_cluster_rule_ids": unknown_cluster_rule_ids,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether single-sample candidates were coarsened into a valid runtime rule tree."
    )
    parser.add_argument(
        "--candidates",
        default="results/unified_rules_3000/semantic_experience_distilled_for_cluster.json",
    )
    parser.add_argument(
        "--generalized",
        default="results/unified_rules_3000/semantic_experience_generalized.json",
    )
    parser.add_argument(
        "--formal",
        default="results/unified_rules_3000/semantic_experience_generalized_for_cluster.json",
    )
    parser.add_argument("--catalog", default="catalogs/rules_unified_3000.json")
    parser.add_argument(
        "--output",
        default="results/unified_rules_3000/rule_coarsening_audit.json",
    )
    parser.add_argument("--residual-bucket-limit", type=int, default=12)
    args = parser.parse_args()

    report = audit_rule_coarsening(
        candidates=_load_json(Path(args.candidates)),
        generalized=_load_json(Path(args.generalized)),
        formal=_load_json(Path(args.formal)),
        catalog=_load_json(Path(args.catalog)),
        residual_bucket_limit=args.residual_bucket_limit,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
