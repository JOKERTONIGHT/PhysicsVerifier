from __future__ import annotations

from typing import Any, Dict, List, Set

from core.rule_catalog_retrieval import iter_rule_leaves

from .models import ValidationResult


def validate_catalog(catalog: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    seen_rule_ids: Set[str] = set()

    domains = catalog.get("domains") if isinstance(catalog, dict) else None
    if not isinstance(domains, list):
        return ValidationResult(ok=False, errors=["catalog.domains must be a list"])

    for domain in domains:
        if not isinstance(domain, dict):
            errors.append("domain entry must be an object")
            continue
        domain_name = str(domain.get("name") or "Unknown")
        topics = domain.get("topics")
        if not isinstance(topics, list):
            errors.append(f"{domain_name}: topics must be a list")
            continue
        for topic in topics:
            if not isinstance(topic, dict):
                errors.append(f"{domain_name}: topic entry must be an object")
                continue
            topic_name = str(topic.get("name") or "Unknown")
            rules = list(iter_rule_leaves(topic))
            for rule in rules:
                rid = str(rule.get("rule_id") or rule.get("id") or "").strip()
                if not rid:
                    errors.append(f"{domain_name}/{topic_name}: rule missing rule_id")
                    continue
                if rid in seen_rule_ids:
                    errors.append(f"duplicate rule_id: {rid}")
                seen_rule_ids.add(rid)
                if not isinstance(rule.get("support"), dict):
                    warnings.append(f"{rid}: missing support object")
                if not isinstance(rule.get("symbolic_hint"), dict):
                    warnings.append(f"{rid}: missing symbolic_hint object")
                profile = str(rule.get("precision_profile") or "").strip().lower()
                if profile and profile not in {"strict", "balanced", "recall"}:
                    warnings.append(f"{rid}: invalid precision_profile {profile!r}")
                if not rule.get("preconditions"):
                    warnings.append(f"{rid}: missing precision preconditions")
                if not rule.get("violation_signatures"):
                    warnings.append(f"{rid}: missing precision violation_signatures")
                if not rule.get("evidence_requirements"):
                    warnings.append(f"{rid}: missing precision evidence_requirements")
                policy = str(rule.get("symbolic_policy") or "").strip().lower()
                if policy and policy not in {"suppress_on_pass", "suppress_on_inconclusive", "require_fail"}:
                    warnings.append(f"{rid}: invalid symbolic_policy {policy!r}")
                features = rule.get("match_features") if isinstance(rule.get("match_features"), dict) else {}
                genericish = set(str(x).strip().lower() for x in (features.get("trigger_keywords") or []))
                genericish.update(str(x).strip().lower() for x in (features.get("object_keywords") or []))
                if genericish and genericish.issubset({"energy", "force", "equation", "formula", "calculation", "result"}):
                    warnings.append(f"{rid}: only generic match signals; mark publishable=false or add precision metadata")

            cluster_rule_ids = set()
            for cluster in topic.get("clusters") or []:
                if not isinstance(cluster, dict):
                    continue
                for rid in cluster.get("rule_ids") or []:
                    cluster_rule_ids.add(str(rid))
            topic_rule_ids = {str(rule.get("rule_id") or rule.get("id") or "") for rule in rules}
            dangling = cluster_rule_ids.difference(topic_rule_ids)
            for rid in sorted(dangling):
                errors.append(f"{domain_name}/{topic_name}: cluster references missing rule_id {rid}")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
