from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _contains_cjk(text: str) -> bool:
    src = _norm_text(text)
    return bool(re.search(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", src))


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = _norm_text(text)
    if not raw:
        raise RuntimeError("Cluster refinement model returned empty content.")
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.S | re.I)
    if fenced:
        data = json.loads(fenced.group(1))
        if isinstance(data, dict):
            return data
    loose = re.search(r"\{.*\}", raw, flags=re.S)
    if loose:
        data = json.loads(loose.group(0))
        if isinstance(data, dict):
            return data
    raise RuntimeError(f"Cluster refinement model did not return valid JSON. Preview: {raw[:300]}")


def _assert_english_only(raw: Dict[str, Any]) -> None:
    fields = [
        _norm_text(raw.get("topic_summary") or ""),
        _norm_text(raw.get("rationale") or ""),
    ]
    for cluster in raw.get("clusters", []) or []:
        if not isinstance(cluster, dict):
            continue
        fields.extend(
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
    offenders = [item for item in fields if _contains_cjk(item)]
    if offenders:
        raise RuntimeError(f"Cluster refinement must be English-only. Offending preview: {offenders[0][:120]}")


def _build_client(*, api_key: str, base_url: str | None, trust_env: bool) -> Any:
    if not OpenAI:
        raise RuntimeError("OpenAI package is not available.")
    kwargs: Dict[str, Any] = {"api_key": api_key, "base_url": base_url}
    if httpx is not None:
        kwargs["http_client"] = httpx.Client(trust_env=trust_env)
    return OpenAI(**kwargs)


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
    _assert_english_only(parsed)
    return parsed


def _normalize_refined_cluster_proposal(raw: Dict[str, Any], valid_rule_ids: set[str]) -> Dict[str, Any]:
    clusters: List[Dict[str, Any]] = []
    used_rule_ids: set[str] = set()
    for item in raw.get("clusters", []) or []:
        if not isinstance(item, dict):
            continue
        candidate_rule_ids = [
            rid for rid in _ordered_unique(item.get("candidate_rule_ids") or []) if rid in valid_rule_ids
        ]
        if not candidate_rule_ids:
            continue
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
    residual_rule_ids = [
        rid for rid in _ordered_unique(raw.get("residual_rule_ids") or []) if rid in valid_rule_ids and rid not in used_rule_ids
    ]
    return {
        "topic_summary": _norm_text(raw.get("topic_summary") or ""),
        "rationale": _norm_text(raw.get("rationale") or ""),
        "clusters": clusters,
        "residual_rule_ids": residual_rule_ids,
    }


def refine_cluster_proposals(
    *,
    draft_payload: Dict[str, Any],
    client: Any,
    model: str,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    proposals = [item for item in (draft_payload.get("proposals") or []) if isinstance(item, dict)]
    refined: List[Dict[str, Any]] = []
    system_prompt = (
        "You are refining scenario-cluster drafts for a physics rule tree. Preserve topic coverage, improve cluster "
        "boundaries, and make the output builder-ready. Every concrete rule_id must be assigned to exactly one cluster "
        "or left in residual_rule_ids for fallback general reasoning. Avoid empty clusters, avoid duplicate rule "
        "assignments, and keep generated text in English only. Return JSON only."
    )
    for proposal in proposals:
        prompt_payload = {
            "domain": proposal.get("domain"),
            "topic": proposal.get("topic"),
            "topic_key": proposal.get("topic_key"),
            "draft_topic_summary": proposal.get("topic_summary"),
            "draft_rationale": proposal.get("rationale"),
            "draft_clusters": proposal.get("clusters") or [],
            "draft_residual_rule_ids": proposal.get("residual_rule_ids") or [],
            "output_schema": {
                "topic_summary": "string",
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
        valid_rule_ids = {
            rid
            for cluster in (proposal.get("clusters") or [])
            if isinstance(cluster, dict)
            for rid in (cluster.get("candidate_rule_ids") or [])
        }
        valid_rule_ids.update(_ordered_unique(proposal.get("residual_rule_ids") or []))
        raw = _chat_json(
            client,
            model=model,
            temperature=temperature,
            system_prompt=system_prompt,
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        )
        normalized = _normalize_refined_cluster_proposal(raw, valid_rule_ids)
        refined.append(
            {
                "domain": proposal.get("domain"),
                "topic": proposal.get("topic"),
                "topic_key": proposal.get("topic_key"),
                **normalized,
            }
        )
    return {
        "metadata": {
            "generator": "cluster_blueprint_refinement_v1",
            "model": model,
            "topic_count": len(refined),
        },
        "proposals": refined,
    }


def build_generated_blueprints_from_refined_proposals(refined_payload: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for proposal in refined_payload.get("proposals", []) or []:
        if not isinstance(proposal, dict):
            continue
        topic_key = _norm_text(proposal.get("topic_key") or "").casefold()
        if not topic_key:
            continue
        topic_clusters: List[Dict[str, Any]] = []
        for cluster in proposal.get("clusters", []) or []:
            if not isinstance(cluster, dict):
                continue
            rule_ids = _ordered_unique(cluster.get("candidate_rule_ids") or [])
            if not rule_ids:
                continue
            cluster_id = _norm_text(cluster.get("cluster_id") or "")
            cluster_name = _norm_text(cluster.get("name") or "")
            topic_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "name": cluster_name,
                    "description": _norm_text(cluster.get("description") or cluster.get("summary") or ""),
                    "includes": _ordered_unique(cluster.get("includes") or []),
                    "excludes": _ordered_unique(cluster.get("excludes") or []),
                    "entry_cues": _ordered_unique(cluster.get("entry_cues") or []),
                    "related_clusters": _ordered_unique(cluster.get("related_clusters") or []),
                    "rule_groups": [
                        {
                            "group_id": f"{cluster_id}_rules",
                            "name": f"{cluster_name} Rules".strip(),
                            "summary": _norm_text(cluster.get("summary") or cluster.get("description") or ""),
                            "activation_condition": _norm_text(cluster.get("summary") or cluster.get("description") or ""),
                            "rule_ids": rule_ids,
                        }
                    ],
                }
            )
        residual_rule_ids = _ordered_unique(proposal.get("residual_rule_ids") or [])
        if residual_rule_ids:
            topic_clusters.append(
                {
                    "cluster_id": "general_reasoning",
                    "name": "General Topic Reasoning",
                    "description": "Fallback cluster for residual topic rules kept outside the specific scenario buckets.",
                    "includes": ["topic-specific residual checks"],
                    "excludes": [],
                    "entry_cues": [],
                    "related_clusters": [],
                    "rule_groups": [
                        {
                            "group_id": "general_reasoning_checks",
                            "name": "General Topic Reasoning Checks",
                            "summary": "Residual topic-specific checks outside the stronger scenario buckets.",
                            "activation_condition": "Use only if the problem belongs to the topic but not to a more specific cluster.",
                            "rule_ids": residual_rule_ids,
                        }
                    ],
                }
            )
        out[topic_key] = topic_clusters
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine draft cluster proposals and emit generated blueprints.")
    parser.add_argument("--draft", type=str, required=True, help="Path to draft proposal payload.")
    parser.add_argument("--output", type=str, required=True, help="Path to write refined proposal payload.")
    parser.add_argument("--generated-blueprints-output", type=str, default=None, help="Optional path for builder-ready generated blueprints.")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--trust-env", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or ""
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    draft_payload = _load_json(Path(args.draft))
    client = _build_client(
        api_key=api_key,
        base_url=args.base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None,
        trust_env=bool(args.trust_env),
    )
    refined = refine_cluster_proposals(
        draft_payload=draft_payload,
        client=client,
        model=_norm_text(args.model),
        temperature=float(args.temperature),
    )
    _dump_json(Path(args.output), refined)
    if args.generated_blueprints_output:
        generated = build_generated_blueprints_from_refined_proposals(refined)
        _dump_json(Path(args.generated_blueprints_output), generated)


if __name__ == "__main__":
    main()
