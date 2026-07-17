from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional

from core.rule_catalog_retrieval import norm_text, ordered_unique

try:
    import httpx
except ImportError:  # pragma: no cover - environment-dependent
    httpx = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - environment-dependent
    OpenAI = None  # type: ignore[assignment]


class SemanticSelectionError(RuntimeError):
    """Identify which semantic-tree stage failed without hiding the root error."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = str(stage or "unknown")
        self.cause = cause
        super().__init__(f"{self.stage}: {type(cause).__name__}: {cause}")


class UnifiedSemanticMatcher:
    MAX_SELECTED_DOMAINS = 2
    MAX_SELECTED_TOPICS = 3
    MAX_SELECTED_CLUSTERS = 4
    MAX_SELECTED_RULES = 5
    RULE_CANDIDATE_BATCH_SIZE = 24
    RULE_CANDIDATE_BATCH_CHARS = 24_000
    INPUT_POLICY = "background_navigation_prediction_rule_only"

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        trust_env: bool | None = None,
        max_selected_rules: int | None = None,
        rule_candidate_batch_size: int | None = None,
        rule_candidate_batch_chars: int | None = None,
    ) -> None:
        self.model = norm_text(model)
        self.temperature = float(temperature)
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or None
        env_trust = norm_text(os.getenv("UNIFIED_SEMANTIC_TRUST_ENV") or "")
        self.trust_env = bool(trust_env) if trust_env is not None else env_trust in {"1", "true", "yes", "on"}
        self.max_selected_rules = max(
            1,
            int(max_selected_rules if max_selected_rules is not None else self.MAX_SELECTED_RULES),
        )
        self.rule_candidate_batch_size = max(
            1,
            int(
                rule_candidate_batch_size
                if rule_candidate_batch_size is not None
                else self.RULE_CANDIDATE_BATCH_SIZE
            ),
        )
        self.rule_candidate_batch_chars = max(
            2_048,
            int(
                rule_candidate_batch_chars
                if rule_candidate_batch_chars is not None
                else self.RULE_CANDIDATE_BATCH_CHARS
            ),
        )
        self._client = client

    def _refresh_env_config(self) -> None:
        # SemanticRuleChecker may load .env before this matcher is first used.
        # Re-read only missing values so explicit constructor arguments still win.
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY") or None
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or None

    @property
    def available(self) -> bool:
        if self._client is not None:
            return True
        self._refresh_env_config()
        return bool(OpenAI and self.model and self.api_key)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        self._refresh_env_config()
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

    @staticmethod
    def _extract_json_object(text: str, *, list_key: str | None = None) -> Dict[str, Any]:
        raw = norm_text(text)
        if not raw:
            raise RuntimeError("Semantic matcher returned empty content.")
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
            raw = re.sub(r"\s*```$", "", raw).strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
            if list_key and isinstance(parsed, list):
                return {list_key: parsed}
        except Exception:
            pass
        fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, flags=re.S | re.I)
        if fenced:
            try:
                parsed = json.loads(fenced.group(1))
                if isinstance(parsed, dict):
                    return parsed
                if list_key and isinstance(parsed, list):
                    return {list_key: parsed}
            except Exception:
                pass
        loose = re.search(r"\{.*\}", raw, flags=re.S)
        if loose:
            try:
                parsed = json.loads(loose.group(0))
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
        if list_key:
            loose_list = re.search(r"\[.*\]", raw, flags=re.S)
            if loose_list:
                try:
                    parsed = json.loads(loose_list.group(0))
                    if isinstance(parsed, list):
                        return {list_key: parsed}
                except Exception:
                    pass
        preview = raw[:300]
        hint = ""
        if raw.startswith("{") and not raw.endswith("}"):
            hint = " The response looks truncated."
        expected = f" or a JSON array for '{list_key}'" if list_key else ""
        raise RuntimeError(f"Semantic matcher must return a JSON object{expected}.{hint} Preview: {preview}")

    def _chat_json(self, *, system_prompt: str, user_prompt: str, list_key: str | None = None) -> Dict[str, Any]:
        client = self._get_client()
        request: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if norm_text(os.getenv("OPENAI_DISABLE_THINKING") or "") in {"1", "true", "yes", "on"}:
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        response = client.chat.completions.create(
            **request,
        )
        content = norm_text(response.choices[0].message.content if response.choices else "")
        return self._extract_json_object(content, list_key=list_key)

    @staticmethod
    def _problem_background(sample: Dict[str, Any]) -> str:
        return "\n".join(
            [
                f"Question:\n{norm_text(sample.get('question') or '')}",
                f"Context:\n{norm_text(sample.get('context') or '')}",
            ]
        ).strip()

    @staticmethod
    def _student_solution(sample: Dict[str, Any]) -> str:
        return norm_text(sample.get("prediction") or "")

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
                    "summary": norm_text(domain.get("summary") or ""),
                    "topic_count": len(topics),
                    "sample_topics": topic_names[:5],
                }
            )
        return out

    @staticmethod
    def _build_topic_candidates(catalog: Dict[str, Any], domains: Iterable[str]) -> List[Dict[str, Any]]:
        domain_filter = {norm_text(item) for item in domains if norm_text(item)}
        if not domain_filter:
            return []
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
                out.append(
                    {
                        "domain": domain_name,
                        "topic": norm_text(topic.get("name") or "Unknown"),
                        "summary": norm_text(topic.get("summary") or ""),
                        "rule_count": len(topic.get("rules") or []),
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
            out.append(
                {
                    "rule_id": norm_text(rule.get("rule_id") or ""),
                    "title": norm_text(rule.get("title") or ""),
                    "summary": norm_text(rule.get("summary") or ""),
                    "trigger": norm_text(rule.get("trigger") or ""),
                    "check_logic": norm_text(rule.get("check_logic") or ""),
                    "error_type": norm_text(rule.get("error_type") or "logic") or "logic",
                    "preconditions": [
                        norm_text(item) for item in (rule.get("preconditions") or []) if norm_text(item)
                    ],
                    "violation_signatures": [
                        norm_text(item) for item in (rule.get("violation_signatures") or []) if norm_text(item)
                    ],
                    "negative_conditions": [
                        norm_text(item) for item in (rule.get("negative_conditions") or []) if norm_text(item)
                    ],
                    "evidence_requirements": [
                        norm_text(item) for item in (rule.get("evidence_requirements") or []) if norm_text(item)
                    ],
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
                        "group_id": norm_text(group.get("id") or group.get("group_id") or ""),
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
                    "cluster_id": norm_text(cluster.get("id") or cluster.get("cluster_id") or ""),
                    "cluster": norm_text(cluster.get("name") or "Unknown"),
                    "summary": norm_text(cluster.get("summary") or ""),
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
        if not domain_candidates:
            return {"domain_judgments": [], "selected_domains": []}
        prompt_payload = {
            "problem_background": self._problem_background(sample),
            "candidate_domains": domain_candidates,
            "max_selected_domains": self.MAX_SELECTED_DOMAINS,
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
                "You are a physics rule navigator. From the problem background only, select the minimum set of "
                "physics domains whose laws are applicable to the stated physical system. Do not infer problem "
                "facts from any student solution. Reject adjacent domains that match only by vocabulary. If the "
                "problem background is incomplete, be conservative. Return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            list_key="domains",
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
        best_by_domain: Dict[str, Dict[str, Any]] = {}
        for judgment in judgments:
            current = best_by_domain.get(judgment["domain"])
            if current is None or float(judgment["score"]) > float(current["score"]):
                best_by_domain[judgment["domain"]] = judgment
        judgments = list(best_by_domain.values())
        judgments.sort(key=lambda item: (-float(item["score"]), item["domain"]))
        judgments = judgments[: self.MAX_SELECTED_DOMAINS]
        return {"domain_judgments": judgments, "selected_domains": [item["domain"] for item in judgments]}

    def select_topics_semantically(self, sample: Dict[str, Any], catalog: Dict[str, Any], domains: Iterable[str]) -> Dict[str, Any]:
        selected_domain_names = [norm_text(item) for item in domains if norm_text(item)]
        if not selected_domain_names:
            return {"topic_judgments": [], "selected_topics": []}
        topic_candidates = self._build_topic_candidates(catalog, selected_domain_names)
        if not topic_candidates:
            return {"topic_judgments": [], "selected_topics": []}
        prompt_payload = {
            "problem_background": self._problem_background(sample),
            "max_selected_topics": self.MAX_SELECTED_TOPICS,
            "candidate_topics": [
                {
                    "domain": item["domain"],
                    "topic": item["topic"],
                    "summary": item["summary"],
                    "rule_count": item["rule_count"],
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
                "You are a physics rule navigator. Inside the selected domains, use only the problem background to "
                "choose the minimum set of topics that govern the physical mechanism. Prefer 1-2 topics. Reject "
                "neighboring concepts, prerequisite knowledge, downstream consequences, and shared-symbol matches. "
                "Do not infer problem facts from a student solution. Treat topic summaries as hard semantic "
                "boundaries. If the background is incomplete, be conservative. Return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            list_key="topics",
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
        best_by_topic: Dict[tuple[str, str], Dict[str, Any]] = {}
        for judgment in judgments:
            key = (judgment["domain"], judgment["topic"])
            current = best_by_topic.get(key)
            if current is None or float(judgment["score"]) > float(current["score"]):
                best_by_topic[key] = judgment
        judgments = list(best_by_topic.values())
        judgments.sort(key=lambda item: (-float(item["score"]), item["domain"], item["topic"]))
        judgments = judgments[: self.MAX_SELECTED_TOPICS]
        return {"topic_judgments": judgments, "selected_topics": judgments}

    def select_clusters_semantically(self, sample: Dict[str, Any], selected_topics: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        all_judgments: List[Dict[str, Any]] = []
        for topic_match in selected_topics:
            cluster_candidates = self._build_cluster_candidates(topic_match)
            if not cluster_candidates:
                continue
            prompt_payload = {
                "problem_background": self._problem_background(sample),
                "max_selected_clusters": self.MAX_SELECTED_CLUSTERS,
                "domain": topic_match["domain"],
                "topic": topic_match["topic"],
                "topic_summary": norm_text(topic_match.get("topic_obj", {}).get("summary") or ""),
                "candidate_clusters": [
                    {
                        "cluster_id": item["cluster_id"],
                        "cluster": item["cluster"],
                        "summary": item["summary"],
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
                    "You are a physics rule navigator. Inside the selected topic, use only the problem background to "
                    "choose the minimum set of scenario clusters that describe the concrete physical mechanism and "
                    "conditions. Prefer one cluster and add another only for a distinct applicable mechanism. Reject "
                    "generic approximations and neighboring derivation styles. Do not infer problem facts from a "
                    "student solution. Respect cluster summaries and activation conditions as hard boundaries. "
                    "Return JSON only."
                ),
                user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
                list_key="clusters",
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
        best_by_cluster: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for judgment in all_judgments:
            key = (judgment["domain"], judgment["topic"], judgment["cluster_id"])
            current = best_by_cluster.get(key)
            if current is None or float(judgment["score"]) > float(current["score"]):
                best_by_cluster[key] = judgment
        all_judgments = list(best_by_cluster.values())
        all_judgments.sort(key=lambda item: (-float(item["score"]), item["domain"], item["topic"], item["cluster_id"]))
        all_judgments = all_judgments[: self.MAX_SELECTED_CLUSTERS]
        return {"cluster_judgments": all_judgments, "selected_clusters": all_judgments}

    @staticmethod
    def _clip_prompt_text(value: Any, max_chars: int) -> str:
        text = norm_text(value or "")
        if len(text) <= max_chars:
            return text
        if max_chars <= 1:
            return text[:max_chars]
        return f"{text[: max_chars - 1]}…"

    @classmethod
    def _clip_prompt_items(cls, values: Iterable[Any], max_chars: int) -> List[str]:
        items = [norm_text(item) for item in values if norm_text(item)][:6]
        if not items:
            return []
        per_item = max(16, max_chars // len(items))
        return [cls._clip_prompt_text(item, per_item) for item in items]

    def _rule_prompt_candidate(self, item: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "rule_id": item["rule_id"],
            "title": item["title"],
            "summary": item["summary"],
            "trigger": item["trigger"],
            "check_logic": item["check_logic"],
            "error_type": item["error_type"],
            "preconditions": item.get("preconditions") or [],
            "violation_signatures": item.get("violation_signatures") or [],
            "negative_conditions": item.get("negative_conditions") or [],
            "evidence_requirements": item.get("evidence_requirements") or [],
            "symbolic_hint": item["symbolic_hint"],
        }
        candidate_limit = self.rule_candidate_batch_chars - 2
        if len(json.dumps(payload, ensure_ascii=False, separators=(",", ":"))) <= candidate_limit:
            return payload

        available = max(256, candidate_limit - 512)
        symbolic_hint = item.get("symbolic_hint") if isinstance(item.get("symbolic_hint"), dict) else {}
        compact = {
            "rule_id": item["rule_id"],
            "title": self._clip_prompt_text(item["title"], max(32, int(available * 0.04))),
            "summary": self._clip_prompt_text(item["summary"], max(64, int(available * 0.10))),
            "trigger": self._clip_prompt_text(item["trigger"], max(64, int(available * 0.10))),
            "check_logic": self._clip_prompt_text(item["check_logic"], max(64, int(available * 0.15))),
            "error_type": self._clip_prompt_text(item["error_type"], 100),
            "preconditions": self._clip_prompt_items(
                item.get("preconditions") or [], max(64, int(available * 0.07))
            ),
            "violation_signatures": self._clip_prompt_items(
                item.get("violation_signatures") or [], max(64, int(available * 0.07))
            ),
            "negative_conditions": self._clip_prompt_items(
                item.get("negative_conditions") or [], max(64, int(available * 0.07))
            ),
            "evidence_requirements": self._clip_prompt_items(
                item.get("evidence_requirements") or [], max(64, int(available * 0.07))
            ),
            "symbolic_hint": {
                "primitive": self._clip_prompt_text(symbolic_hint.get("primitive") or "", 100),
                "canonical": self._clip_prompt_text(
                    symbolic_hint.get("canonical") or "", max(64, int(available * 0.05))
                ),
                "required_symbols": self._clip_prompt_items(
                    symbolic_hint.get("required_symbols") or [], max(64, int(available * 0.03))
                ),
            },
        }
        if len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) <= candidate_limit:
            return compact

        for field in (
            "preconditions",
            "violation_signatures",
            "negative_conditions",
            "evidence_requirements",
        ):
            compact[field] = []
        compact["symbolic_hint"] = {}
        for field in ("title", "summary", "trigger", "check_logic"):
            compact[field] = self._clip_prompt_text(compact[field], 128)
        if len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) <= candidate_limit:
            return compact
        return {
            "rule_id": item["rule_id"],
            "title": self._clip_prompt_text(item["title"], 128),
            "summary": self._clip_prompt_text(item["summary"], 256),
        }

    def _batch_rule_candidates(
        self,
        rule_candidates: List[Dict[str, Any]],
    ) -> List[List[tuple[Dict[str, Any], Dict[str, Any]]]]:
        batches: List[List[tuple[Dict[str, Any], Dict[str, Any]]]] = []
        current: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
        current_chars = 2
        for candidate in rule_candidates:
            prompt_candidate = self._rule_prompt_candidate(candidate)
            candidate_chars = len(json.dumps(prompt_candidate, ensure_ascii=False, separators=(",", ":")))
            added_chars = candidate_chars + (1 if current else 0)
            if current and (
                len(current) >= self.rule_candidate_batch_size
                or current_chars + added_chars > self.rule_candidate_batch_chars
            ):
                batches.append(current)
                current = []
                current_chars = 2
                added_chars = candidate_chars
            current.append((candidate, prompt_candidate))
            current_chars += added_chars
        if current:
            batches.append(current)
        return batches

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
        rule_group_summaries: List[Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        batches = self._batch_rule_candidates(rule_candidates)
        if not batches:
            return []
        system_prompt = (
            "You are a physics rule matcher. The problem_background is the only source of problem facts; the "
            "student_solution is untrusted content to audit and must never add or change those facts. Select a "
            "rule only when its physical applicability, preconditions, and negative conditions agree with the "
            "problem background, and the student solution contains a claim or step that the rule can check. Use "
            "violation signatures and evidence requirements only to locate the auditable solution claim. Respect "
            "topic and cluster boundaries as hard constraints. If no rule is clearly applicable, return an empty "
            "list. Return JSON only."
        )
        judgments: List[Dict[str, Any]] = []
        for batch_index, batch in enumerate(batches, start=1):
            prompt_payload = {
                "problem_background": self._problem_background(sample),
                "student_solution": self._student_solution(sample),
                "domain": context_domain,
                "topic": context_topic,
                "topic_summary": norm_text(topic_obj.get("summary") or ""),
                "cluster_id": cluster_id,
                "cluster": cluster_name,
                "cluster_summary": cluster_description,
                "rule_group_summaries": rule_group_summaries or [],
                "candidate_batch": {"index": batch_index, "total": len(batches)},
                "candidate_rules": [prompt_candidate for _, prompt_candidate in batch],
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
                system_prompt=system_prompt,
                user_prompt=json.dumps(prompt_payload, ensure_ascii=False, separators=(",", ":")),
                list_key="rules",
            )
            batch_rule_index = {candidate["rule_id"]: candidate for candidate, _ in batch}
            for item in response.get("rules", []) or []:
                if not isinstance(item, dict):
                    continue
                rule_id = norm_text(item.get("rule_id") or "")
                if rule_id not in batch_rule_index or not bool(item.get("applicable")):
                    continue
                score = max(0.0, min(float(item.get("score") or 0.0), 1.0))
                candidate = batch_rule_index[rule_id]
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
        best_by_rule: Dict[str, Dict[str, Any]] = {}
        for judgment in judgments:
            current = best_by_rule.get(judgment["rule_id"])
            if current is None or float(judgment["score"]) > float(current["score"]):
                best_by_rule[judgment["rule_id"]] = judgment
        merged = list(best_by_rule.values())
        merged.sort(key=lambda item: (-float(item["score"]), item["rule_id"]))
        return merged

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
                rule_candidates.append(
                    {
                        "rule_id": norm_text(rule.get("rule_id") or ""),
                        "title": norm_text(rule.get("title") or ""),
                        "summary": norm_text(rule.get("summary") or ""),
                        "trigger": norm_text(rule.get("trigger") or ""),
                        "check_logic": norm_text(rule.get("check_logic") or ""),
                        "error_type": norm_text(rule.get("error_type") or "logic") or "logic",
                        "preconditions": [
                            norm_text(x) for x in (rule.get("preconditions") or []) if norm_text(x)
                        ],
                        "violation_signatures": [
                            norm_text(x) for x in (rule.get("violation_signatures") or []) if norm_text(x)
                        ],
                        "negative_conditions": [
                            norm_text(x) for x in (rule.get("negative_conditions") or []) if norm_text(x)
                        ],
                        "evidence_requirements": [
                            norm_text(x) for x in (rule.get("evidence_requirements") or []) if norm_text(x)
                        ],
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
                cluster_description=norm_text(item.get("cluster_obj", {}).get("summary") or ""),
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
        globally_capped = capped[: self.max_selected_rules]
        return {"rule_judgments": globally_capped, "selected_rules": globally_capped}

    def _tree_result(
        self,
        *,
        domain_result: Dict[str, Any],
        topic_result: Dict[str, Any] | None = None,
        cluster_result: Dict[str, Any] | None = None,
        rule_result: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        topics = topic_result or {"topic_judgments": [], "selected_topics": []}
        clusters = cluster_result or {"cluster_judgments": [], "selected_clusters": []}
        rules = rule_result or {"rule_judgments": [], "selected_rules": []}
        return {
            "input_policy": self.INPUT_POLICY,
            "domain_judgments": domain_result["domain_judgments"],
            "topic_judgments": topics["topic_judgments"],
            "cluster_judgments": clusters["cluster_judgments"],
            "rule_judgments": rules["rule_judgments"],
            "selected_domains": domain_result["selected_domains"],
            "selected_topics": topics["selected_topics"],
            "selected_clusters": clusters["selected_clusters"],
            "selected_rules": rules["selected_rules"],
        }

    def select_tree_semantically(self, sample: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
        try:
            domain_result = self.select_domains_semantically(sample, catalog)
        except Exception as exc:
            raise SemanticSelectionError("domain", exc) from exc
        if not domain_result["selected_domains"]:
            return self._tree_result(domain_result=domain_result)
        try:
            topic_result = self.select_topics_semantically(sample, catalog, domain_result["selected_domains"])
        except Exception as exc:
            raise SemanticSelectionError("topic", exc) from exc
        if not topic_result["selected_topics"]:
            return self._tree_result(domain_result=domain_result, topic_result=topic_result)
        try:
            cluster_result = self.select_clusters_semantically(sample, topic_result["selected_topics"])
        except Exception as exc:
            raise SemanticSelectionError("cluster", exc) from exc
        try:
            rule_result = self.select_rules_semantically(
                sample,
                topic_result["selected_topics"],
                cluster_result["selected_clusters"],
            )
        except Exception as exc:
            raise SemanticSelectionError("rule", exc) from exc
        return self._tree_result(
            domain_result=domain_result,
            topic_result=topic_result,
            cluster_result=cluster_result,
            rule_result=rule_result,
        )
