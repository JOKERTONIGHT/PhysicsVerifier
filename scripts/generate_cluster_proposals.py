from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import httpx
except ImportError:  # pragma: no cover - environment dependent
    httpx = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment dependent
    OpenAI = None  # type: ignore[assignment]


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _ordered_unique(items: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for item in items:
        text = _norm_text(item)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _topic_key(domain: str, topic: str) -> str:
    return f"{_norm_text(domain).lower()}::{_norm_text(topic).lower()}"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_client(*, api_key: str, base_url: str | None, trust_env: bool) -> Any:
    if not OpenAI:
        raise RuntimeError("OpenAI package is not available.")
    kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if httpx is not None:
        kwargs["http_client"] = httpx.Client(trust_env=trust_env)
    return OpenAI(**kwargs)


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = _norm_text(text)
    if not raw:
        raise RuntimeError("Cluster proposal model returned empty content.")

    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S | re.I)
    if fenced:
        try:
            data = json.loads(fenced.group(1))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    loose = re.search(r"\{.*\}", raw, flags=re.S)
    if loose:
        try:
            data = json.loads(loose.group(0))
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    preview = raw[:300]
    raise RuntimeError(f"Cluster proposal model did not return a valid JSON object. Preview: {preview}")


def _contains_cjk(text: str) -> bool:
    src = _norm_text(text)
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", src))


def _assert_english_only_proposal(raw: Dict[str, Any]) -> None:
    fields_to_check: List[str] = [
        _norm_text(raw.get("topic_summary") or ""),
        _norm_text(raw.get("rationale") or ""),
    ]
    for cluster in raw.get("clusters", []) or []:
        if not isinstance(cluster, dict):
            continue
        fields_to_check.extend(
            [
                _norm_text(cluster.get("cluster_id") or ""),
                _norm_text(cluster.get("name") or ""),
                _norm_text(cluster.get("summary") or ""),
                _norm_text(cluster.get("description") or ""),
                *[_norm_text(item) for item in (cluster.get("includes") or [])],
                *[_norm_text(item) for item in (cluster.get("excludes") or [])],
                *[_norm_text(item) for item in (cluster.get("entry_cues") or [])],
                *[_norm_text(item) for item in (cluster.get("related_clusters") or [])],
            ]
        )
    offenders = [text for text in fields_to_check if _contains_cjk(text)]
    if offenders:
        raise RuntimeError(
            "Cluster proposal must be English-only for all generated semantic fields. "
            f"Offending preview: {offenders[0][:120]}"
        )


def _chat_json(client: Any, *, model: str, temperature: float, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content if response.choices else ""
    parsed = _extract_json_object(_norm_text(content))
    _assert_english_only_proposal(parsed)
    if not isinstance(parsed, dict):
        raise RuntimeError("Cluster proposal model must return a JSON object.")
    return parsed


def _parse_csv_values(raw_values: Sequence[str] | None) -> set[str]:
    out: set[str] = set()
    for raw in raw_values or []:
        for part in str(raw).split(","):
            text = _norm_text(part)
            if text:
                out.add(text)
    return out


def _collect_topic_candidates(
    catalog: Dict[str, Any],
    *,
    only_missing_clusters: bool,
    domain_filters: set[str],
    topic_filters: set[str],
    max_topics: int,
    min_rule_count: int,
) -> List[Dict[str, Any]]:
    topics: List[Dict[str, Any]] = []
    for domain in catalog.get("domains", []) or []:
        if not isinstance(domain, dict):
            continue
        domain_name = _norm_text(domain.get("name") or "Unknown")
        if domain_filters and domain_name not in domain_filters:
            continue
        for topic in domain.get("topics", []) or []:
            if not isinstance(topic, dict):
                continue
            topic_name = _norm_text(topic.get("name") or "Unknown")
            if topic_filters and topic_name not in topic_filters and _topic_key(domain_name, topic_name) not in topic_filters:
                continue
            rules = [item for item in (topic.get("rules") or []) if isinstance(item, dict)]
            scenario_clusters = topic.get("scenario_clusters") or []
            if only_missing_clusters and scenario_clusters:
                continue
            if len(rules) < min_rule_count:
                continue
            topics.append(
                {
                    "domain": domain_name,
                    "topic": topic_name,
                    "topic_obj": topic,
                    "rule_count": len(rules),
                    "has_clusters": bool(scenario_clusters),
                }
            )
    topics.sort(key=lambda item: (-int(item["rule_count"]), item["domain"], item["topic"]))
    if max_topics > 0:
        topics = topics[:max_topics]
    return topics


def _build_topic_prompt_payload(topic_match: Dict[str, Any]) -> Dict[str, Any]:
    topic = topic_match["topic_obj"]
    rules = [item for item in (topic.get("rules") or []) if isinstance(item, dict)]
    return {
        "domain": topic_match["domain"],
        "topic": topic_match["topic"],
        "topic_description": _norm_text(topic.get("description") or ""),
        "topic_includes": _ordered_unique(topic.get("includes") or []),
        "topic_excludes": _ordered_unique(topic.get("excludes") or []),
        "related_topics": _ordered_unique(topic.get("related_topics") or []),
        "existing_scenario_clusters": [
            {
                "cluster_id": _norm_text(cluster.get("cluster_id") or ""),
                "name": _norm_text(cluster.get("name") or ""),
                "description": _norm_text(cluster.get("description") or ""),
                "rule_ids": _ordered_unique(cluster.get("rule_ids") or []),
            }
            for cluster in (topic.get("scenario_clusters") or [])
            if isinstance(cluster, dict)
        ],
        "rules": [
            {
                "rule_id": _norm_text(rule.get("rule_id") or ""),
                "title": _norm_text(rule.get("title") or ""),
                "trigger": _norm_text(rule.get("trigger") or ""),
                "check_logic": _norm_text(rule.get("check_logic") or ""),
                "error_type": _norm_text(rule.get("error_type") or ""),
                "scope": _norm_text(rule.get("scope") or ""),
                "support_count": int((rule.get("support") or {}).get("count") or 0),
                "sample_ids": _ordered_unique((rule.get("support") or {}).get("sample_ids") or [])[:5],
            }
            for rule in rules
        ],
        "output_schema": {
            "topic_summary": "string",
            "should_add_clusters": True,
            "rationale": "string",
            "clusters": [
                {
                    "cluster_id": "string",
                    "name": "string",
                    "summary": "string",
                    "description": "string",
                    "includes": ["string"],
                    "excludes": ["string"],
                    "entry_cues": ["string"],
                    "related_clusters": ["string"],
                    "candidate_rule_ids": ["string"],
                }
            ],
            "residual_rule_ids": ["string"],
        },
    }


def _normalize_cluster_proposal(raw: Dict[str, Any], valid_rule_ids: set[str]) -> Dict[str, Any]:
    clusters: List[Dict[str, Any]] = []
    used_rule_ids: set[str] = set()
    for item in raw.get("clusters", []) or []:
        if not isinstance(item, dict):
            continue
        candidate_rule_ids = [rid for rid in _ordered_unique(item.get("candidate_rule_ids") or []) if rid in valid_rule_ids]
        used_rule_ids.update(candidate_rule_ids)
        clusters.append(
            {
                "cluster_id": _norm_text(item.get("cluster_id") or ""),
                "name": _norm_text(item.get("name") or ""),
                "summary": _norm_text(item.get("summary") or ""),
                "description": _norm_text(item.get("description") or ""),
                "includes": _ordered_unique(item.get("includes") or []),
                "excludes": _ordered_unique(item.get("excludes") or []),
                "entry_cues": _ordered_unique(item.get("entry_cues") or []),
                "related_clusters": _ordered_unique(item.get("related_clusters") or []),
                "candidate_rule_ids": candidate_rule_ids,
            }
        )
    residual_rule_ids = [rid for rid in _ordered_unique(raw.get("residual_rule_ids") or []) if rid in valid_rule_ids and rid not in used_rule_ids]
    return {
        "topic_summary": _norm_text(raw.get("topic_summary") or ""),
        "should_add_clusters": bool(raw.get("should_add_clusters")),
        "rationale": _norm_text(raw.get("rationale") or ""),
        "clusters": clusters,
        "residual_rule_ids": residual_rule_ids,
    }


def generate_cluster_proposals(
    *,
    catalog: Dict[str, Any],
    client: Any,
    model: str,
    temperature: float,
    only_missing_clusters: bool,
    domain_filters: set[str],
    topic_filters: set[str],
    max_topics: int,
    min_rule_count: int,
) -> Dict[str, Any]:
    topic_candidates = _collect_topic_candidates(
        catalog,
        only_missing_clusters=only_missing_clusters,
        domain_filters=domain_filters,
        topic_filters=topic_filters,
        max_topics=max_topics,
        min_rule_count=min_rule_count,
    )
    proposals: List[Dict[str, Any]] = []
    system_prompt = (
        "You are designing scenario clusters for a physics rule tree. Group rules by distinct audit scenario or "
        "failure mode, not by superficial keyword overlap and not by textbook chapter names. Prefer 2-4 clusters. "
        "Each cluster must represent a materially different reasoning scene, have clear semantic boundaries, and map "
        "to concrete rule_ids. Avoid over-fragmentation. If a topic is too small or too homogeneous to justify "
        "clusters, set should_add_clusters to false. Use English only for every generated field, including cluster "
        "names, summaries, descriptions, includes, excludes, entry_cues, and rationale. Do not output Chinese or "
        "mixed-language phrases. Return JSON only."
    )
    for topic_match in topic_candidates:
        payload = _build_topic_prompt_payload(topic_match)
        raw = _chat_json(
            client,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            user_prompt=json.dumps(payload, ensure_ascii=False, indent=2),
        )
        valid_rule_ids = {item["rule_id"] for item in payload["rules"]}
        normalized = _normalize_cluster_proposal(raw, valid_rule_ids)
        proposals.append(
            {
                "domain": topic_match["domain"],
                "topic": topic_match["topic"],
                "topic_key": _topic_key(topic_match["domain"], topic_match["topic"]),
                "rule_count": topic_match["rule_count"],
                "existing_cluster_count": len(topic_match["topic_obj"].get("scenario_clusters") or []),
                **normalized,
            }
        )
    return {
        "metadata": {
            "generator": "semantic_cluster_proposal_v1",
            "model": model,
            "only_missing_clusters": only_missing_clusters,
            "topic_count": len(proposals),
            "min_rule_count": min_rule_count,
        },
        "proposals": proposals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scenario-cluster proposals from the current unified rules catalog.")
    parser.add_argument("--catalog", type=str, default="catalogs/rules_unified.json")
    parser.add_argument("--output", type=str, default="results/cluster_proposals.json")
    parser.add_argument("--model", type=str, default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--trust-env", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--domains", action="append", default=[], help="Repeat or comma-separate domain filters.")
    parser.add_argument("--topics", action="append", default=[], help="Repeat or comma-separate topic filters. Topic key format domain::topic is also accepted.")
    parser.add_argument("--max-topics", type=int, default=12)
    parser.add_argument("--min-rule-count", type=int, default=1)
    parser.add_argument("--all-topics", action="store_true", help="Include topics that already have scenario clusters.")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    catalog = _load_json(Path(args.catalog))
    client = _build_client(
        api_key=api_key,
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None,
        trust_env=bool(args.trust_env),
    )
    result = generate_cluster_proposals(
        catalog=catalog,
        client=client,
        model=_norm_text(args.model),
        temperature=float(args.temperature),
        only_missing_clusters=not bool(args.all_topics),
        domain_filters=_parse_csv_values(args.domains),
        topic_filters=_parse_csv_values(args.topics),
        max_topics=int(args.max_topics),
        min_rule_count=int(args.min_rule_count),
    )
    _dump_json(Path(args.output), result)
    print(f"Wrote cluster proposals to {args.output}")


if __name__ == "__main__":
    main()
