from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _semantic_samples(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        return [item for item in payload["samples"] if isinstance(item, dict)]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _topic_key(topic_guess: Any) -> str:
    if not isinstance(topic_guess, dict):
        return "Unknown::Unknown"
    domain = str(topic_guess.get("domain") or "Unknown")
    topic = str(topic_guess.get("topic") or "Unknown")
    return f"{domain}::{topic}"


def _is_unknown_topic(topic_guess: Any) -> bool:
    if not isinstance(topic_guess, dict):
        return True
    domain = str(topic_guess.get("domain") or "").strip()
    topic = str(topic_guess.get("topic") or "").strip()
    return not domain or not topic or domain == "Unknown" or topic == "Unknown"


def _is_failure_placeholder(item: Dict[str, Any]) -> bool:
    audit = item.get("semantic_audit") if isinstance(item.get("semantic_audit"), dict) else {}
    text_parts = [str(audit.get("summary") or "")]
    for err in audit.get("key_errors") or []:
        if isinstance(err, dict):
            text_parts.append(str(err.get("message") or ""))
            text_parts.append(str(err.get("evidence") or ""))
    joined = "\n".join(text_parts)
    return "LLM调用失败" in joined or "LLM call failed" in joined


def _has_list(aux: Dict[str, Any], key: str) -> bool:
    return isinstance(aux.get(key), list) and bool(aux.get(key))


def _has_text(aux: Dict[str, Any], key: str) -> bool:
    return bool(str(aux.get(key) or "").strip())


def analyze_run(
    *,
    semantic_path: Path,
    distilled_path: Path,
    expected_samples: int = 0,
    output_path: Path | None = None,
    strict: bool = False,
) -> Dict[str, Any]:
    samples = _semantic_samples(_load_json(semantic_path))
    distilled = _load_json(distilled_path)
    rules = [item for item in (distilled.get("rules") if isinstance(distilled, dict) else []) or [] if isinstance(item, dict)]

    sample_ids = [str(item.get("sample_id") or "") for item in samples]
    id_counts = Counter(sample_ids)
    duplicate_ids = sorted([sid for sid, count in id_counts.items() if sid and count > 1])
    missing_required = 0
    topic_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"sample_count": 0, "experience_rule_count": 0})

    failure_count = 0
    empty_rule_count = 0
    unknown_topic_count = 0
    for item in samples:
        if not item.get("sample_id") or not isinstance(item.get("topic_guess"), dict) or not isinstance(item.get("experience_rules"), list):
            missing_required += 1
        if _is_failure_placeholder(item):
            failure_count += 1
        rules_for_sample = item.get("experience_rules") if isinstance(item.get("experience_rules"), list) else []
        if not rules_for_sample:
            empty_rule_count += 1
        if _is_unknown_topic(item.get("topic_guess")):
            unknown_topic_count += 1
        key = _topic_key(item.get("topic_guess"))
        topic_stats[key]["sample_count"] += 1
        topic_stats[key]["experience_rule_count"] += len(rules_for_sample)

    distilled_topic_keys = {
        f"{str(rule.get('domain') or 'Unknown')}::{str(rule.get('topic') or 'Unknown')}"
        for rule in rules
    }
    unknown_rule_count = sum(
        1
        for rule in rules
        if str(rule.get("domain") or "") in {"", "Unknown"} or str(rule.get("topic") or "") in {"", "Unknown"}
    )

    auxiliary_stats = {
        "rules_with_auxiliary": 0,
        "rules_with_node_summary": 0,
        "rules_with_scene_cues": 0,
        "rules_with_boundary_cues": 0,
        "rules_with_explore_cues": 0,
        "rules_with_evidence_sample_ids": 0,
    }
    for rule in rules:
        aux = rule.get("auxiliary") if isinstance(rule.get("auxiliary"), dict) else {}
        has_any = (
            _has_text(aux, "node_summary")
            or _has_list(aux, "scene_cues")
            or _has_list(aux, "boundary_cues")
            or _has_list(aux, "explore_cues")
            or _has_list(aux, "evidence_sample_ids")
        )
        if has_any:
            auxiliary_stats["rules_with_auxiliary"] += 1
        if _has_text(aux, "node_summary"):
            auxiliary_stats["rules_with_node_summary"] += 1
        if _has_list(aux, "scene_cues"):
            auxiliary_stats["rules_with_scene_cues"] += 1
        if _has_list(aux, "boundary_cues"):
            auxiliary_stats["rules_with_boundary_cues"] += 1
        if _has_list(aux, "explore_cues"):
            auxiliary_stats["rules_with_explore_cues"] += 1
        if _has_list(aux, "evidence_sample_ids"):
            auxiliary_stats["rules_with_evidence_sample_ids"] += 1

    report: Dict[str, Any] = {
        "semantic": {
            "path": str(semantic_path),
            "sample_count": len(samples),
            "expected_samples": expected_samples,
            "duplicate_sample_id_count": len(duplicate_ids),
            "duplicate_sample_ids": duplicate_ids[:20],
            "missing_required_field_count": missing_required,
            "failure_placeholder_count": failure_count,
            "empty_rule_sample_count": empty_rule_count,
            "unknown_topic_sample_count": unknown_topic_count,
            "topics": dict(sorted(topic_stats.items())),
        },
        "distilled": {
            "path": str(distilled_path),
            "total_rules": len(rules),
            "topic_bucket_count": len(distilled_topic_keys),
            "unknown_rule_count": unknown_rule_count,
            "auxiliary": auxiliary_stats,
        },
        "strict_failures": [],
    }

    if expected_samples and len(samples) != expected_samples:
        report["strict_failures"].append("sample_count_mismatch")
    if duplicate_ids:
        report["strict_failures"].append("duplicate_sample_ids")
    if missing_required:
        report["strict_failures"].append("missing_required_fields")
    if failure_count:
        report["strict_failures"].append("failure_placeholders")

    if output_path:
        _write_json(output_path, report)
    if strict and report["strict_failures"]:
        raise SystemExit("; ".join(report["strict_failures"]))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze semantic experience run outputs.")
    parser.add_argument("--semantic", required=True)
    parser.add_argument("--distilled", required=True)
    parser.add_argument("--expected-samples", type=int, default=0)
    parser.add_argument("--output", default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = analyze_run(
        semantic_path=Path(args.semantic),
        distilled_path=Path(args.distilled),
        expected_samples=args.expected_samples,
        output_path=Path(args.output) if args.output else None,
        strict=args.strict,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
