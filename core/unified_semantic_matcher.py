from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Optional

from core.unified_retrieval import norm_text, ordered_unique

try:
    import httpx
except ImportError:  # pragma: no cover - environment-dependent
    httpx = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment-dependent
    OpenAI = None  # type: ignore[assignment]


class UnifiedSemanticMatcher:
    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        trust_env: bool | None = None,
    ) -> None:
        self.model = norm_text(model)
        self.temperature = float(temperature)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or None
        env_trust = norm_text(os.getenv("UNIFIED_SEMANTIC_TRUST_ENV") or "")
        self.trust_env = bool(trust_env) if trust_env is not None else env_trust in {"1", "true", "yes", "on"}
        self._client = client

    @property
    def available(self) -> bool:
        if self._client is not None:
            return True
        return bool(OpenAI and self.model and self.api_key)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not OpenAI:
            raise RuntimeError("OpenAI package is not available.")
        if not self.model:
            raise RuntimeError("Semantic matcher model is not configured.")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured.")
        client_kwargs: Dict[str, Any] = {"api_key": self.api_key, "base_url": self.base_url}
        if httpx is not None:
            client_kwargs["http_client"] = httpx.Client(trust_env=self.trust_env)
        self._client = OpenAI(**client_kwargs)
        return self._client

    def _chat_json(self, *, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = norm_text(response.choices[0].message.content if response.choices else "")
        if not content:
            raise RuntimeError("Semantic matcher returned empty content.")
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise RuntimeError("Semantic matcher must return a JSON object.")
        return parsed

    @staticmethod
    def _sample_text(sample: Dict[str, Any]) -> str:
        return "\n".join(
            [
                f"Question:\n{norm_text(sample.get('question') or '')}",
                f"Context:\n{norm_text(sample.get('context') or '')}",
                f"Prediction:\n{norm_text(sample.get('prediction') or '')}",
            ]
        ).strip()

    @staticmethod
    def _build_domain_candidates(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for domain in catalog.get("domains", []) or []:
            if not isinstance(domain, dict):
                continue
            topics = [topic for topic in (domain.get("topics") or []) if isinstance(topic, dict)]
            topic_names = [norm_text(topic.get("name") or "") for topic in topics if norm_text(topic.get("name") or "")]
            out.append(
                {
                    "domain": norm_text(domain.get("name") or "Unknown"),
                    "topic_count": len(topics),
                    "sample_topics": topic_names[:5],
                }
            )
        return out

    @staticmethod
    def _build_topic_candidates(catalog: Dict[str, Any], domains: Iterable[str]) -> List[Dict[str, Any]]:
        domain_filter = {norm_text(item) for item in domains if norm_text(item)}
        out: List[Dict[str, Any]] = []
        for domain in catalog.get("domains", []) or []:
            if not isinstance(domain, dict):
                continue
            domain_name = norm_text(domain.get("name") or "Unknown")
            if domain_filter and domain_name not in domain_filter:
                continue
            for topic in domain.get("topics", []) or []:
                if not isinstance(topic, dict):
                    continue
                retrieval_hints = topic.get("retrieval_hints") if isinstance(topic.get("retrieval_hints"), dict) else {}
                topic_keywords = ordered_unique(retrieval_hints.get("topic_keywords") or [])[:5]
                scene_keywords = ordered_unique(retrieval_hints.get("scene_keywords") or [])[:4]
                out.append(
                    {
                        "domain": domain_name,
                        "topic": norm_text(topic.get("name") or "Unknown"),
                        "description": norm_text(topic.get("description") or ""),
                        "includes": ordered_unique(topic.get("includes") or [])[:5],
                        "excludes": ordered_unique(topic.get("excludes") or [])[:5],
                        "related_topics": ordered_unique(topic.get("related_topics") or [])[:5],
                        "rule_count": len(topic.get("rules") or []),
                        "topic_keywords": topic_keywords,
                        "scene_keywords": scene_keywords,
                        "topic_obj": topic,
                    }
                )
        return out

    @staticmethod
    def _build_rule_candidates(topic_match: Dict[str, Any]) -> List[Dict[str, Any]]:
        topic_obj = topic_match.get("topic_obj") if isinstance(topic_match.get("topic_obj"), dict) else {}
        out: List[Dict[str, Any]] = []
        for rule in topic_obj.get("rules", []) or []:
            if not isinstance(rule, dict):
                continue
            support = rule.get("support") if isinstance(rule.get("support"), dict) else {}
            out.append(
                {
                    "rule_id": norm_text(rule.get("rule_id") or ""),
                    "title": norm_text(rule.get("title") or ""),
                    "trigger": norm_text(rule.get("trigger") or ""),
                    "check_logic": norm_text(rule.get("check_logic") or ""),
                    "scope": norm_text(rule.get("scope") or "domain") or "domain",
                    "support_count": int(support.get("count") or 0),
                    "symbolic_hint": rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {},
                    "rule_obj": rule,
                }
            )
        return out

    @staticmethod
    def _build_cluster_candidates(topic_match: Dict[str, Any]) -> List[Dict[str, Any]]:
        topic_obj = topic_match.get("topic_obj") if isinstance(topic_match.get("topic_obj"), dict) else {}
        topic_rules = {
            str(rule.get("rule_id") or ""): rule
            for rule in (topic_obj.get("rules") or [])
            if isinstance(rule, dict) and norm_text(rule.get("rule_id") or "")
        }
        out: List[Dict[str, Any]] = []
        for cluster in topic_obj.get("scenario_clusters", []) or []:
            if not isinstance(cluster, dict):
                continue
            cluster_rule_ids = [norm_text(item) for item in (cluster.get("rule_ids") or []) if norm_text(item)]
            rule_groups = []
            for group in cluster.get("rule_groups", []) or []:
                if not isinstance(group, dict):
                    continue
                rule_groups.append(
                    {
                        "group_id": norm_text(group.get("group_id") or ""),
                        "name": norm_text(group.get("name") or ""),
                        "summary": norm_text(group.get("summary") or ""),
                        "activation_condition": norm_text(group.get("activation_condition") or ""),
                        "rule_ids": [norm_text(item) for item in (group.get("rule_ids") or []) if norm_text(item)],
                    }
                )
            out.append(
                {
                    "domain": topic_match["domain"],
                    "topic": topic_match["topic"],
                    "cluster_id": norm_text(cluster.get("cluster_id") or ""),
                    "cluster": norm_text(cluster.get("name") or "Unknown"),
                    "description": norm_text(cluster.get("description") or ""),
                    "includes": ordered_unique(cluster.get("includes") or []),
                    "excludes": ordered_unique(cluster.get("excludes") or []),
                    "entry_cues": ordered_unique(cluster.get("entry_cues") or []),
                    "related_clusters": ordered_unique(cluster.get("related_clusters") or []),
                    "rule_groups": rule_groups,
                    "rule_ids": cluster_rule_ids,
                    "topic_obj": topic_obj,
                    "cluster_obj": cluster,
                    "topic_rules": topic_rules,
                }
            )
        return out

    def select_domains_semantically(self, sample: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
        domain_candidates = self._build_domain_candidates(catalog)
        prompt_payload = {
            "sample": self._sample_text(sample),
            "candidate_domains": domain_candidates,
            "output_schema": {
                "domains": [
                    {
                        "domain": "string",
                        "relevant": True,
                        "score": 0.0,
                        "reason": "short reason",
                    }
                ]
            },
        }
        response = self._chat_json(
            system_prompt=(
                "You are a physics rule matcher. Select the minimum set of physics domains that are directly "
                "necessary for auditing the student's solution. Keep domains only if they provide an independent "
                "error-diagnosis lens. Reject domains that are merely background knowledge, adjacent subject matter, "
                "or weakly related by vocabulary. If uncertain, exclude rather than include. Return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        )
        judgments: List[Dict[str, Any]] = []
        valid_domains = {item["domain"] for item in domain_candidates}
        for item in response.get("domains", []) or []:
            if not isinstance(item, dict):
                continue
            domain = norm_text(item.get("domain") or "")
            if domain not in valid_domains:
                continue
            relevant = bool(item.get("relevant"))
            score = max(0.0, min(float(item.get("score") or 0.0), 1.0))
            if not relevant:
                continue
            judgments.append(
                {
                    "domain": domain,
                    "relevant": True,
                    "score": score,
                    "reason": norm_text(item.get("reason") or ""),
                }
            )
        judgments.sort(key=lambda item: (-float(item["score"]), item["domain"]))
        return {"domain_judgments": judgments, "selected_domains": [item["domain"] for item in judgments]}

    def select_topics_semantically(self, sample: Dict[str, Any], catalog: Dict[str, Any], domains: Iterable[str]) -> Dict[str, Any]:
        topic_candidates = self._build_topic_candidates(catalog, domains)
        prompt_payload = {
            "sample": self._sample_text(sample),
            "candidate_topics": [
                {
                    "domain": item["domain"],
                    "topic": item["topic"],
                    "description": item["description"],
                    "includes": item["includes"],
                    "excludes": item["excludes"],
                    "related_topics": item["related_topics"],
                    "rule_count": item["rule_count"],
                    "topic_keywords": item["topic_keywords"],
                    "scene_keywords": item["scene_keywords"],
                }
                for item in topic_candidates
            ],
            "output_schema": {
                "topics": [
                    {
                        "domain": "string",
                        "topic": "string",
                        "relevant": True,
                        "score": 0.0,
                        "reason": "short reason",
                    }
                ]
            },
        }
        response = self._chat_json(
            system_prompt=(
                "You are a physics rule matcher. Inside the provided domains, select only the minimum set of topics "
                "that are directly necessary for auditing the student's solution. Prefer 1-2 topics when possible. "
                "Add extra topics only when they provide a clearly distinct audit path that cannot be covered by the "
                "stronger topics already selected. Reject topics that are merely neighboring concepts, prerequisite "
                "knowledge, downstream consequences, or weakly related by shared symbols or vocabulary. When one "
                "topic is mechanism-specific and another is only a generic bookkeeping lens such as energy/accounting/"
                "consistency, keep the mechanism-specific topic and reject the generic one unless it contributes an "
                "independent error mode. Use the topic description, includes, and excludes as hard semantic "
                "boundaries. If uncertain, exclude rather than include. Return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        )
        candidate_index = {(item["domain"], item["topic"]): item for item in topic_candidates}
        judgments: List[Dict[str, Any]] = []
        for item in response.get("topics", []) or []:
            if not isinstance(item, dict):
                continue
            domain = norm_text(item.get("domain") or "")
            topic = norm_text(item.get("topic") or "")
            key = (domain, topic)
            if key not in candidate_index or not bool(item.get("relevant")):
                continue
            score = max(0.0, min(float(item.get("score") or 0.0), 1.0))
            candidate = candidate_index[key]
            judgments.append(
                {
                    "domain": domain,
                    "topic": topic,
                    "relevant": True,
                    "score": score,
                    "reason": norm_text(item.get("reason") or ""),
                    "topic_obj": candidate["topic_obj"],
                }
            )
        judgments.sort(key=lambda item: (-float(item["score"]), item["domain"], item["topic"]))
        return {"topic_judgments": judgments, "selected_topics": judgments}

    def select_clusters_semantically(self, sample: Dict[str, Any], selected_topics: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        all_judgments: List[Dict[str, Any]] = []
        for topic_match in selected_topics:
            cluster_candidates = self._build_cluster_candidates(topic_match)
            if not cluster_candidates:
                continue
            prompt_payload = {
                "sample": self._sample_text(sample),
                "domain": topic_match["domain"],
                "topic": topic_match["topic"],
                "topic_description": norm_text(topic_match.get("topic_obj", {}).get("description") or ""),
                "candidate_clusters": [
                    {
                        "cluster_id": item["cluster_id"],
                        "cluster": item["cluster"],
                        "description": item["description"],
                        "includes": item["includes"],
                        "excludes": item["excludes"],
                        "entry_cues": item["entry_cues"],
                        "related_clusters": item["related_clusters"],
                        "rule_group_summaries": [
                            {
                                "group_id": group["group_id"],
                                "name": group["name"],
                                "summary": group["summary"],
                                "activation_condition": group["activation_condition"],
                                "rule_count": len(group["rule_ids"]),
                            }
                            for group in item["rule_groups"]
                        ],
                        "rule_count": len(item["rule_ids"]),
                    }
                    for item in cluster_candidates
                ],
                "output_schema": {
                    "clusters": [
                        {
                            "cluster_id": "string",
                            "relevant": True,
                            "score": 0.0,
                            "reason": "short reason",
                        }
                    ]
                },
            }
            response = self._chat_json(
                system_prompt=(
                    "You are a physics rule matcher. Inside the provided topic, select only the minimum set of "
                    "scenario clusters that are directly necessary for auditing the student's solution. Prefer 1 "
                    "cluster when possible, and select a second cluster only if it represents a clearly distinct "
                    "failure mode. Reject clusters that are generic approximations, neighboring derivation styles, "
                    "or merely weakly related through vocabulary. When one cluster captures the concrete physical "
                    "mechanism and another is only a generic accounting/consistency lens, keep the mechanism cluster "
                    "and reject the generic one unless it exposes an independent failure mode. Respect the topic and "
                    "cluster includes/excludes as hard boundaries. If uncertain, exclude rather than include. Return "
                    "JSON only."
                ),
                user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            )
            cluster_index = {item["cluster_id"]: item for item in cluster_candidates}
            topic_judgments: List[Dict[str, Any]] = []
            for item in response.get("clusters", []) or []:
                if not isinstance(item, dict):
                    continue
                cluster_id = norm_text(item.get("cluster_id") or "")
                if cluster_id not in cluster_index or not bool(item.get("relevant")):
                    continue
                score = max(0.0, min(float(item.get("score") or 0.0), 1.0))
                candidate = cluster_index[cluster_id]
                topic_judgments.append(
                    {
                        "cluster_id": cluster_id,
                        "cluster": candidate["cluster"],
                        "domain": candidate["domain"],
                        "topic": candidate["topic"],
                        "relevant": True,
                        "score": score,
                        "reason": norm_text(item.get("reason") or ""),
                        "cluster_obj": candidate["cluster_obj"],
                        "topic_obj": candidate["topic_obj"],
                        "rule_groups": candidate["rule_groups"],
                        "rule_ids": candidate["rule_ids"],
                        "topic_rules": candidate["topic_rules"],
                    }
                )
            topic_judgments.sort(key=lambda item: (-float(item["score"]), item["cluster_id"]))
            all_judgments.extend(topic_judgments)
        all_judgments.sort(key=lambda item: (-float(item["score"]), item["domain"], item["topic"], item["cluster_id"]))
        return {"cluster_judgments": all_judgments, "selected_clusters": all_judgments}

    def _select_rules_for_context(
        self,
        *,
        sample: Dict[str, Any],
        context_domain: str,
        context_topic: str,
        topic_obj: Dict[str, Any],
        rule_candidates: List[Dict[str, Any]],
        cluster_id: str = "",
        cluster_name: str = "",
        cluster_description: str = "",
        cluster_includes: List[str] | None = None,
        cluster_excludes: List[str] | None = None,
        rule_group_summaries: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        prompt_payload = {
            "sample": self._sample_text(sample),
            "domain": context_domain,
            "topic": context_topic,
            "topic_description": norm_text(topic_obj.get("description") or ""),
            "topic_includes": ordered_unique(topic_obj.get("includes") or []),
            "topic_excludes": ordered_unique(topic_obj.get("excludes") or []),
            "cluster_id": cluster_id,
            "cluster": cluster_name,
            "cluster_description": cluster_description,
            "cluster_includes": ordered_unique(cluster_includes or []),
            "cluster_excludes": ordered_unique(cluster_excludes or []),
            "rule_group_summaries": rule_group_summaries or [],
            "candidate_rules": [
                {
                    "rule_id": item["rule_id"],
                    "title": item["title"],
                    "trigger": item["trigger"],
                    "check_logic": item["check_logic"],
                    "scope": item["scope"],
                    "support_count": item["support_count"],
                    "symbolic_hint": item["symbolic_hint"],
                }
                for item in rule_candidates
            ],
            "output_schema": {
                "rules": [
                    {
                        "rule_id": "string",
                        "applicable": True,
                        "score": 0.0,
                        "reason": "short reason",
                    }
                ]
            },
        }
        response = self._chat_json(
            system_prompt=(
                "You are a physics rule matcher. Select only the rules that are actually applicable to auditing the "
                "student's solution. Use the topic and cluster boundaries as hard constraints. Reject rules that are "
                "generic approximations, neighboring derivation styles, or merely tangentially related. If no "
                "cluster is active for the topic, be conservative and keep at most the 1-2 strongest topic-level "
                "rules. If no rule is clearly applicable, return an empty list. Return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
        )
        rule_index = {item["rule_id"]: item for item in rule_candidates}
        judgments: List[Dict[str, Any]] = []
        for item in response.get("rules", []) or []:
            if not isinstance(item, dict):
                continue
            rule_id = norm_text(item.get("rule_id") or "")
            if rule_id not in rule_index or not bool(item.get("applicable")):
                continue
            score = max(0.0, min(float(item.get("score") or 0.0), 1.0))
            candidate = rule_index[rule_id]
            judgments.append(
                {
                    "rule_id": rule_id,
                    "title": candidate["title"],
                    "domain": context_domain,
                    "topic": context_topic,
                    "cluster_id": cluster_id,
                    "cluster": cluster_name,
                    "applicable": True,
                    "score": score,
                    "reason": norm_text(item.get("reason") or ""),
                    "rule_obj": candidate["rule_obj"],
                }
            )
        judgments.sort(key=lambda item: (-float(item["score"]), item["rule_id"]))
        return judgments

    def select_rules_semantically(
        self,
        sample: Dict[str, Any],
        selected_topics: Iterable[Dict[str, Any]],
        selected_clusters: Iterable[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        all_judgments: List[Dict[str, Any]] = []
        selected_topic_list = list(selected_topics)
        topic_index = {(item["domain"], item["topic"]): item for item in selected_topic_list}
        cluster_list = list(selected_clusters or [])
        topics_with_rule_hits: set[tuple[str, str]] = set()
        topics_with_cluster_attempts: set[tuple[str, str]] = set()

        for item in cluster_list:
            topic_key = (item["domain"], item["topic"])
            topics_with_cluster_attempts.add(topic_key)
            rule_candidates = []
            topic_rules = item.get("topic_rules") if isinstance(item.get("topic_rules"), dict) else {}
            for rule_id in item.get("rule_ids", []) or []:
                rule = topic_rules.get(rule_id)
                if not isinstance(rule, dict):
                    continue
                support = rule.get("support") if isinstance(rule.get("support"), dict) else {}
                rule_candidates.append(
                    {
                        "rule_id": norm_text(rule.get("rule_id") or ""),
                        "title": norm_text(rule.get("title") or ""),
                        "trigger": norm_text(rule.get("trigger") or ""),
                        "check_logic": norm_text(rule.get("check_logic") or ""),
                        "scope": norm_text(rule.get("scope") or "domain") or "domain",
                        "support_count": int(support.get("count") or 0),
                        "symbolic_hint": rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {},
                        "rule_obj": rule,
                    }
                )
            if not rule_candidates:
                continue
            topic_obj = item.get("topic_obj") if isinstance(item.get("topic_obj"), dict) else {}
            cluster_judgments = self._select_rules_for_context(
                sample=sample,
                context_domain=item["domain"],
                context_topic=item["topic"],
                topic_obj=topic_obj,
                rule_candidates=rule_candidates,
                cluster_id=item["cluster_id"],
                cluster_name=item["cluster"],
                cluster_description=norm_text(item.get("cluster_obj", {}).get("description") or ""),
                cluster_includes=item.get("cluster_obj", {}).get("includes") or [],
                cluster_excludes=item.get("cluster_obj", {}).get("excludes") or [],
                rule_group_summaries=[
                    {
                        "group_id": group["group_id"],
                        "name": group["name"],
                        "summary": group["summary"],
                        "activation_condition": group["activation_condition"],
                        "rule_count": len(group["rule_ids"]),
                    }
                    for group in item.get("rule_groups", [])
                ],
            )
            if cluster_judgments:
                topics_with_rule_hits.add(topic_key)
            all_judgments.extend(cluster_judgments)

        fallback_topics: List[Dict[str, Any]] = []
        if not cluster_list:
            fallback_topics = selected_topic_list
        else:
            for topic_match in selected_topic_list:
                topic_key = (topic_match["domain"], topic_match["topic"])
                if topic_key not in topics_with_rule_hits:
                    fallback_topics.append(topic_match)

        for topic_match in fallback_topics:
            rule_candidates = self._build_rule_candidates(topic_match)
            if not rule_candidates:
                continue
            topic_judgments = self._select_rules_for_context(
                sample=sample,
                context_domain=topic_match["domain"],
                context_topic=topic_match["topic"],
                topic_obj=topic_match.get("topic_obj") if isinstance(topic_match.get("topic_obj"), dict) else {},
                rule_candidates=rule_candidates,
            )
            all_judgments.extend(topic_judgments)

        best_by_rule: Dict[str, Dict[str, Any]] = {}
        for judgment in all_judgments:
            rule_id = judgment["rule_id"]
            current = best_by_rule.get(rule_id)
            if current is None or float(judgment["score"]) > float(current["score"]):
                best_by_rule[rule_id] = judgment
        merged = list(best_by_rule.values())
        merged.sort(key=lambda item: (-float(item["score"]), item["domain"], item["topic"], item["rule_id"]))
        clustered_topic_keys = {(item["domain"], item["topic"]) for item in cluster_list}
        clusterless_kept: Dict[tuple[str, str], int] = {}
        capped: List[Dict[str, Any]] = []
        for judgment in merged:
            topic_key = (judgment["domain"], judgment["topic"])
            if judgment["cluster_id"] or topic_key in clustered_topic_keys:
                capped.append(judgment)
                continue
            kept = clusterless_kept.get(topic_key, 0)
            if kept >= 2:
                continue
            clusterless_kept[topic_key] = kept + 1
            capped.append(judgment)
        return {"rule_judgments": capped, "selected_rules": capped}

    def select_tree_semantically(self, sample: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
        domain_result = self.select_domains_semantically(sample, catalog)
        topic_result = self.select_topics_semantically(sample, catalog, domain_result["selected_domains"])
        cluster_result = self.select_clusters_semantically(sample, topic_result["selected_topics"])
        rule_result = self.select_rules_semantically(sample, topic_result["selected_topics"], cluster_result["selected_clusters"])
        return {
            "domain_judgments": domain_result["domain_judgments"],
            "topic_judgments": topic_result["topic_judgments"],
            "cluster_judgments": cluster_result["cluster_judgments"],
            "rule_judgments": rule_result["rule_judgments"],
            "selected_domains": domain_result["selected_domains"],
            "selected_topics": topic_result["selected_topics"],
            "selected_clusters": cluster_result["selected_clusters"],
            "selected_rules": rule_result["selected_rules"],
        }
