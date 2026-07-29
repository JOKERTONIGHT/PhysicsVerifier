from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return data


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _ordered_unique(values: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        item = _text(value)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _rules(payload: Dict[str, Any], *, source: str) -> List[Dict[str, Any]]:
    rules = payload.get("rules")
    if not isinstance(rules, list):
        raise ValueError(f"{source} must contain a 'rules' array.")
    if any(not isinstance(rule, dict) for rule in rules):
        raise ValueError(f"{source} contains a non-object rule.")
    return rules


def _fingerprint(rule: Dict[str, Any]) -> str:
    fields = (
        "domain",
        "topic",
        "title",
        "trigger",
        "check_logic",
        "error_type",
    )
    canonical = "\n".join(_text(rule.get(field)).casefold() for field in fields)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _ensure_rule_id(rule: Dict[str, Any]) -> str:
    rule_id = _text(rule.get("rule_id"))
    if not rule_id:
        rule_id = f"exp_{_fingerprint(rule)[:16]}"
        rule["rule_id"] = rule_id
    return rule_id


def _validate_candidate(rule: Dict[str, Any]) -> None:
    missing = [
        field
        for field in ("domain", "topic", "title", "trigger", "check_logic")
        if not _text(rule.get(field))
    ]
    if missing:
        raise ValueError(
            f"Candidate {_text(rule.get('rule_id')) or '<without id>'} "
            f"is missing required fields: {', '.join(missing)}"
        )


def _merge_provenance(target: Dict[str, Any], source: Dict[str, Any]) -> bool:
    changed = False
    for field in ("sample_ids", "source_rule_ids"):
        merged = _ordered_unique(
            list(target.get(field) or []) + list(source.get(field) or [])
        )
        if merged != list(target.get(field) or []):
            target[field] = merged
            changed = True
    sample_ids = list(target.get("sample_ids") or [])
    old_count = int(target.get("count") or 0)
    source_count = int(source.get("count") or 0)
    new_count = len(sample_ids) if sample_ids else max(old_count, source_count, 1)
    if new_count != old_count:
        target["count"] = new_count
        changed = True
    return changed


def prepare_incremental_candidates(
    *,
    current_payload: Dict[str, Any],
    new_payload: Dict[str, Any],
    formal_payload: Dict[str, Any] | None = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    current_rules = deepcopy(_rules(current_payload, source="current candidate bank"))
    new_rules = deepcopy(_rules(new_payload, source="new candidate batch"))
    formal_rules = (
        _rules(formal_payload, source="formal rule bank")
        if formal_payload is not None
        else []
    )

    by_id: Dict[str, Dict[str, Any]] = {}
    by_fingerprint: Dict[str, Dict[str, Any]] = {}
    for rule in current_rules:
        _validate_candidate(rule)
        rule_id = _ensure_rule_id(rule)
        if rule_id in by_id:
            raise ValueError(f"Duplicate rule_id in current candidate bank: {rule_id}")
        by_id[rule_id] = rule
        by_fingerprint.setdefault(_fingerprint(rule), rule)

    formal_by_fingerprint = {
        _fingerprint(rule): rule
        for rule in formal_rules
        if isinstance(rule, dict)
    }
    added_ids: List[str] = []
    support_updated_ids: List[str] = []
    covered_by_formal: List[Dict[str, str]] = []
    unchanged_duplicate_ids: List[str] = []
    affected_topics: set[tuple[str, str]] = set()

    for candidate in new_rules:
        _validate_candidate(candidate)
        candidate_id = _ensure_rule_id(candidate)
        fingerprint = _fingerprint(candidate)

        same_id = by_id.get(candidate_id)
        if same_id is not None and _fingerprint(same_id) != fingerprint:
            raise ValueError(
                f"rule_id conflict: {candidate_id} refers to different candidate content."
            )

        existing = same_id or by_fingerprint.get(fingerprint)
        if existing is not None:
            existing_id = _ensure_rule_id(existing)
            if _merge_provenance(existing, candidate):
                support_updated_ids.append(existing_id)
                affected_topics.add(
                    (_text(existing.get("domain")), _text(existing.get("topic")))
                )
            else:
                unchanged_duplicate_ids.append(existing_id)
            continue

        formal = formal_by_fingerprint.get(fingerprint)
        if formal is not None:
            covered_by_formal.append(
                {
                    "candidate_rule_id": candidate_id,
                    "formal_rule_id": _text(formal.get("rule_id")),
                }
            )
            continue

        current_rules.append(candidate)
        by_id[candidate_id] = candidate
        by_fingerprint[fingerprint] = candidate
        added_ids.append(candidate_id)
        affected_topics.add(
            (_text(candidate.get("domain")), _text(candidate.get("topic")))
        )

    merged = deepcopy(current_payload)
    merged["rules"] = current_rules
    summary = dict(merged.get("summary") or {})
    summary["rule_count"] = len(current_rules)
    merged["summary"] = summary
    metadata = dict(merged.get("metadata") or {})
    metadata["incremental_candidate_bank"] = True
    merged["metadata"] = metadata

    report = {
        "summary": {
            "current_candidate_count": len(_rules(current_payload, source="current")),
            "new_batch_count": len(new_rules),
            "merged_candidate_count": len(current_rules),
            "added_count": len(added_ids),
            "support_updated_count": len(support_updated_ids),
            "covered_by_formal_count": len(covered_by_formal),
            "unchanged_duplicate_count": len(unchanged_duplicate_ids),
            "affected_topic_count": len(affected_topics),
        },
        "added_candidate_ids": added_ids,
        "support_updated_candidate_ids": support_updated_ids,
        "covered_by_formal": covered_by_formal,
        "unchanged_duplicate_candidate_ids": unchanged_duplicate_ids,
        "affected_topics": [
            {"domain": domain, "topic": topic}
            for domain, topic in sorted(affected_topics)
        ],
        "next_step": (
            "Use the merged candidate bank as semantic_experience_distilled.json, "
            "then rerun prepare-candidates, embedding clustering, generalization, "
            "compare generalized proposals with existing formal rules inside affected "
            "topics, then rerun formal clustering, blueprints, catalog validation, and "
            "retrieval regression. "
            "Embedding caches ensure only new or changed rules call the embedding API."
        ),
    }
    return merged, report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely merge a new extracted-rule batch into the candidate bank."
    )
    parser.add_argument("--current", required=True, help="Current candidate-bank JSON.")
    parser.add_argument("--new", required=True, help="New extracted candidate JSON.")
    parser.add_argument(
        "--formal",
        default=None,
        help="Optional formal-rule JSON used to suppress exact duplicate candidates.",
    )
    parser.add_argument("--output", required=True, help="Merged candidate-bank JSON.")
    parser.add_argument("--report", required=True, help="Incremental merge report JSON.")
    args = parser.parse_args()

    merged, report = prepare_incremental_candidates(
        current_payload=_load_json(Path(args.current)),
        new_payload=_load_json(Path(args.new)),
        formal_payload=_load_json(Path(args.formal)) if args.formal else None,
    )
    _write_json(Path(args.output), merged)
    _write_json(Path(args.report), report)
    print(json.dumps(report["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
