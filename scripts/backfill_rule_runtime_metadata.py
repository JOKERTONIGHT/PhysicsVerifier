from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.rule_catalog_retrieval import (
    build_scene_keywords,
    build_topic_required_symbols,
    classify_rule_scope,
    extract_keywords,
    norm_text,
    ordered_unique,
)
from rule_framework.normalization import build_match_features, safe_symbolic_hint


DEFAULT_NEGATIVE_CONDITIONS = [
    "intermediate value later corrected",
    "rule not required by the problem",
    "equivalent formulation",
]


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _has_items(value: Any) -> bool:
    return isinstance(value, list) and any(norm_text(item) for item in value)


def _text_items(values: Iterable[Any], limit: int) -> List[str]:
    return ordered_unique(str(item) for item in values if norm_text(item))[:limit]


def _rule_signature(rule: Dict[str, Any], *, include_trigger: bool = True) -> str:
    parts = [
        norm_text(rule.get("title") or "").casefold(),
        norm_text(rule.get("check_logic") or rule.get("description") or "").casefold(),
    ]
    if include_trigger:
        parts.insert(1, norm_text(rule.get("trigger") or "").casefold())
    return "\n".join(parts)


def _iter_catalog_rules(catalog: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            for rule in topic.get("rules", []) or []:
                if isinstance(rule, dict):
                    yield rule


def _iter_catalog_topics(catalog: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = norm_text(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            yield domain_name, norm_text(topic.get("name") or "Unknown"), topic


def _topic_key(domain: str, topic: str) -> Tuple[str, str]:
    return norm_text(domain).casefold(), norm_text(topic).casefold()


def _build_reference_topic_index(reference_catalogs: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for catalog in reference_catalogs:
        for domain, topic, topic_obj in _iter_catalog_topics(catalog):
            index.setdefault(_topic_key(domain, topic), topic_obj)
    return index


def _copy_reference_topic_metadata(
    topic: Dict[str, Any],
    reference: Dict[str, Any],
    *,
    overwrite: bool,
) -> Dict[str, bool]:
    changed = {
        "retrieval_hints": False,
        "knowledge_reference": False,
        "tagged_reference": False,
        "includes": False,
        "excludes": False,
        "related_topics": False,
    }
    for field in ("retrieval_hints", "knowledge_reference", "tagged_reference"):
        value = reference.get(field)
        if not isinstance(value, dict) or not value:
            continue
        if overwrite or not isinstance(topic.get(field), dict) or not topic.get(field):
            topic[field] = json.loads(json.dumps(value, ensure_ascii=False))
            changed[field] = True
    for field in ("includes", "excludes", "related_topics"):
        value = reference.get(field)
        if not _has_items(value):
            continue
        if overwrite or not _has_items(topic.get(field)):
            topic[field] = list(value)
            changed[field] = True
    return changed


def _ensure_topic_metadata(topic: Dict[str, Any], *, domain: str, overwrite: bool) -> Dict[str, bool]:
    """Ensure the main verifier has topic-level routing signals.

    The canonical navigation catalog intentionally strips these fields.  The
    runtime-derived catalog must restore them, otherwise upstream topic routing
    falls back almost entirely to matching the topic name.
    """
    changed = {"retrieval_hints": False, "knowledge_reference": False, "tagged_reference": False}
    topic_name = norm_text(topic.get("name") or "Unknown")
    rules = [rule for rule in (topic.get("rules") or []) if isinstance(rule, dict)]
    titles = [norm_text(rule.get("title") or "") for rule in rules]
    triggers = [norm_text(rule.get("trigger") or "") for rule in rules]
    rule_texts = [text for pair in zip(titles, triggers) for text in pair if text]

    hints = dict(topic.get("retrieval_hints") or {}) if isinstance(topic.get("retrieval_hints"), dict) else {}
    if overwrite or not _has_items(hints.get("topic_keywords")):
        hints["topic_keywords"] = extract_keywords([topic_name, *rule_texts], max_keywords=20)
        changed["retrieval_hints"] = True
    if overwrite or not _has_items(hints.get("scene_keywords")):
        hints["scene_keywords"] = build_scene_keywords(
            topic_name=topic_name,
            tagged_titles=titles,
            tagged_aliases=[],
            rule_texts=triggers,
        )
        changed["retrieval_hints"] = True
    if overwrite or not _has_items(hints.get("required_symbols")):
        required_symbols = build_topic_required_symbols(rules)
        if overwrite or required_symbols:
            if required_symbols != list(hints.get("required_symbols") or []):
                hints["required_symbols"] = required_symbols
                changed["retrieval_hints"] = True
    topic["retrieval_hints"] = hints

    if overwrite or not isinstance(topic.get("knowledge_reference"), dict):
        topic["knowledge_reference"] = {
            "rule_ids": [],
            "keywords": extract_keywords([topic_name, *rule_texts], max_keywords=20),
        }
        changed["knowledge_reference"] = True
    if overwrite or not isinstance(topic.get("tagged_reference"), dict):
        topic["tagged_reference"] = {"source_ids": [], "titles": [], "aliases": [], "keywords": []}
        changed["tagged_reference"] = True

    topic.setdefault("includes", [])
    topic.setdefault("excludes", [])
    topic.setdefault("related_topics", [])
    return changed


def _build_reference_index(reference_catalogs: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for catalog in reference_catalogs:
        for rule in _iter_catalog_rules(catalog):
            exact = _rule_signature(rule, include_trigger=True)
            fallback = _rule_signature(rule, include_trigger=False)
            if exact.strip():
                index.setdefault(("exact", exact), rule)
            if fallback.strip():
                index.setdefault(("fallback", fallback), rule)
    return index


def _find_reference_rule(rule: Dict[str, Any], index: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any] | None:
    exact = _rule_signature(rule, include_trigger=True)
    if ("exact", exact) in index:
        return index[("exact", exact)]
    fallback = _rule_signature(rule, include_trigger=False)
    return index.get(("fallback", fallback))


def _copy_reference_metadata(rule: Dict[str, Any], reference: Dict[str, Any], *, overwrite: bool) -> Dict[str, bool]:
    changed = {
        "reference": True,
        "match_features": False,
        "support": False,
        "llm_hints": False,
        "precision": False,
        "source_rule_ids": False,
    }
    for field in ("match_features", "support", "llm_hints"):
        value = reference.get(field)
        if not isinstance(value, dict) or not value:
            continue
        if overwrite or not isinstance(rule.get(field), dict) or not rule.get(field):
            rule[field] = json.loads(json.dumps(value, ensure_ascii=False))
            changed[field] = True
    for field in ("scope", "symbolic_policy", "precision_profile"):
        value = norm_text(reference.get(field) or "")
        if value and (overwrite or not norm_text(rule.get(field) or "")):
            rule[field] = value
            changed["precision"] = changed["precision"] or field in {"symbolic_policy", "precision_profile"}
    for field in ("preconditions", "violation_signatures", "negative_conditions", "evidence_requirements"):
        value = reference.get(field)
        if _has_items(value) and (overwrite or not _has_items(rule.get(field))):
            rule[field] = list(value)
            changed["precision"] = True
    reference_rule_id = norm_text(reference.get("rule_id") or reference.get("id") or "")
    if reference_rule_id:
        source_rule_ids = ordered_unique(list(rule.get("source_rule_ids") or []) + [reference_rule_id])
        if overwrite or source_rule_ids != list(rule.get("source_rule_ids") or []):
            rule["source_rule_ids"] = source_rule_ids
            changed["source_rule_ids"] = True
    return changed


def _default_precision_metadata(rule: Dict[str, Any]) -> Dict[str, Any]:
    features = rule.get("match_features") if isinstance(rule.get("match_features"), dict) else {}
    trigger_keywords = _text_items(features.get("trigger_keywords") or [], 8)
    object_keywords = _text_items(features.get("object_keywords") or [], 8)
    required_symbols = _text_items(features.get("required_symbols") or [], 6)
    symbolic_hint = safe_symbolic_hint(rule.get("symbolic_hint"))
    primitive = norm_text(symbolic_hint.get("primitive") or "none") or "none"
    scope = norm_text(rule.get("scope") or "domain")
    profile = "balanced" if scope == "meta" else "strict"
    symbolic_policy = "suppress_on_inconclusive" if primitive not in {"", "none"} else "suppress_on_pass"

    preconditions = _text_items(trigger_keywords[:4] + object_keywords[:2], 6)
    violation_signatures = _text_items(required_symbols[:3] + object_keywords[:4] + trigger_keywords[:2], 8)
    evidence_requirements = _text_items(required_symbols[:3] + trigger_keywords[:3] + object_keywords[:3], 8)

    return {
        "precision_profile": profile,
        "publishable": True,
        "preconditions": preconditions,
        "violation_signatures": violation_signatures,
        "negative_conditions": list(DEFAULT_NEGATIVE_CONDITIONS),
        "evidence_requirements": evidence_requirements,
        "symbolic_policy": symbolic_policy,
    }


def _ensure_rule_metadata(rule: Dict[str, Any], *, domain: str, topic: str, overwrite: bool) -> Dict[str, bool]:
    changed = {
        "id": False,
        "path": False,
        "scope": False,
        "symbolic_hint": False,
        "match_features": False,
        "support": False,
        "precision": False,
    }

    rid = norm_text(rule.get("rule_id") or rule.get("id") or "")
    if rid and (overwrite or not norm_text(rule.get("id") or "")):
        rule["id"] = rid
        changed["id"] = True
    if rid and (overwrite or not norm_text(rule.get("rule_id") or "")):
        rule["rule_id"] = rid

    if overwrite or not isinstance(rule.get("symbolic_hint"), dict):
        rule["symbolic_hint"] = safe_symbolic_hint(rule.get("symbolic_hint"))
        changed["symbolic_hint"] = True

    title = norm_text(rule.get("title") or rid)
    trigger = norm_text(rule.get("trigger") or "")
    check_logic = norm_text(rule.get("check_logic") or rule.get("description") or "")

    if overwrite or not norm_text(rule.get("scope") or ""):
        rule["scope"] = classify_rule_scope(title=title, trigger=trigger, check_logic=check_logic, rule_id=rid)
        changed["scope"] = True

    if overwrite or not isinstance(rule.get("path"), dict):
        path = dict(rule.get("path") or {})
        path.setdefault("domain", domain)
        path.setdefault("topic", topic)
        path.setdefault("context", norm_text(rule.get("error_type") or "logic") or "logic")
        path.setdefault("cluster", "unclustered")
        rule["path"] = path
        changed["path"] = True

    if overwrite or not isinstance(rule.get("match_features"), dict):
        hint = safe_symbolic_hint(rule.get("symbolic_hint"))
        features = build_match_features(title, trigger, check_logic, hint)
        negative = _text_items(rule.get("negative_conditions") or DEFAULT_NEGATIVE_CONDITIONS, 8)
        features["negative_keywords"] = negative
        rule["match_features"] = features
        changed["match_features"] = True

    if overwrite or not isinstance(rule.get("support"), dict):
        rule["support"] = {"count": 0, "sample_ids": []}
        changed["support"] = True

    defaults = _default_precision_metadata(rule)
    precision_changed = False
    for key, value in defaults.items():
        if key == "publishable":
            if overwrite or "publishable" not in rule:
                rule[key] = value
                precision_changed = True
        elif key in {"precision_profile", "symbolic_policy"}:
            if overwrite or not norm_text(rule.get(key) or ""):
                rule[key] = value
                precision_changed = True
        else:
            if overwrite or not _has_items(rule.get(key)):
                rule[key] = value
                precision_changed = True
    changed["precision"] = precision_changed
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill non-API runtime metadata for unified rule catalogs.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reference-catalog",
        action="append",
        default=[],
        help="Optional enhanced catalog to reuse existing non-API metadata from; can be repeated.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    catalog = _load_json(args.catalog)
    reference_catalogs = [_load_json(path) for path in (args.reference_catalog or [])]
    reference_index = _build_reference_index(reference_catalogs)
    reference_topic_index = _build_reference_topic_index(reference_catalogs)
    counters = {
        "topics": 0,
        "reference_topic_hits": 0,
        "reference_topic_retrieval_hints": 0,
        "reference_topic_knowledge_reference": 0,
        "reference_topic_tagged_reference": 0,
        "generated_topic_retrieval_hints": 0,
        "generated_topic_knowledge_reference": 0,
        "generated_topic_tagged_reference": 0,
        "rules": 0,
        "reference_hits": 0,
        "reference_match_features": 0,
        "reference_support": 0,
        "reference_llm_hints": 0,
        "reference_precision": 0,
        "reference_source_rule_ids": 0,
        "id": 0,
        "path": 0,
        "scope": 0,
        "symbolic_hint": 0,
        "match_features": 0,
        "support": 0,
        "precision": 0,
    }

    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = norm_text(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = norm_text(topic.get("name") or "Unknown")
            counters["topics"] += 1
            reference_topic = reference_topic_index.get(_topic_key(domain_name, topic_name))
            if reference_topic:
                counters["reference_topic_hits"] += 1
                topic_changed = _copy_reference_topic_metadata(topic, reference_topic, overwrite=args.overwrite)
                for field in ("retrieval_hints", "knowledge_reference", "tagged_reference"):
                    if topic_changed[field]:
                        counters[f"reference_topic_{field}"] += 1
            for rule in topic.get("rules", []) or []:
                if not isinstance(rule, dict):
                    continue
                counters["rules"] += 1
                reference = _find_reference_rule(rule, reference_index) if reference_index else None
                if reference:
                    ref_changed = _copy_reference_metadata(rule, reference, overwrite=args.overwrite)
                    counters["reference_hits"] += 1
                    if ref_changed["match_features"]:
                        counters["reference_match_features"] += 1
                    if ref_changed["support"]:
                        counters["reference_support"] += 1
                    if ref_changed["llm_hints"]:
                        counters["reference_llm_hints"] += 1
                    if ref_changed["precision"]:
                        counters["reference_precision"] += 1
                    if ref_changed["source_rule_ids"]:
                        counters["reference_source_rule_ids"] += 1
                changed = _ensure_rule_metadata(rule, domain=domain_name, topic=topic_name, overwrite=args.overwrite)
                for key, did_change in changed.items():
                    if did_change:
                        counters[key] += 1
            topic_changed = _ensure_topic_metadata(topic, domain=domain_name, overwrite=args.overwrite)
            for field, did_change in topic_changed.items():
                if did_change:
                    counters[f"generated_topic_{field}"] += 1

    metadata = catalog.setdefault("metadata", {})
    metadata["runtime_metadata_backfilled"] = True
    metadata["runtime_metadata_backfill_counts"] = counters
    metadata["runtime_metadata_backfill_note"] = (
        "Runtime-derived metadata with restored topic routing signals and deterministic rule precision fields. "
        "Use LLM enrichment only after an evaluated FP whitelist is available."
    )

    _write_json(args.output, catalog)
    print(json.dumps(counters, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
