from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Callable, Dict, Iterable, List, Optional

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

    def __init__(
        self,
        stage: str,
        cause: Exception,
        *,
        trace: Dict[str, Any] | None = None,
        partial_result: Dict[str, Any] | None = None,
    ) -> None:
        self.stage = str(stage or "unknown")
        self.cause = cause
        self.trace = copy.deepcopy(trace or {})
        self.partial_result = copy.deepcopy(partial_result or {})
        super().__init__(f"{self.stage}: {type(cause).__name__}: {cause}")


class UnifiedSemanticMatcher:
    MAX_SELECTED_DOMAINS = 2
    MAX_SELECTED_TOPICS = 3
    MAX_SELECTED_CLUSTERS = 4
    MAX_SELECTED_RULES = 5
    RULE_CANDIDATE_BATCH_SIZE = 24
    RULE_CANDIDATE_BATCH_CHARS = 24_000
    MAX_JSON_RETRIES = 3
    MAX_RESPONSE_TOKENS = 2_048
    HARD_MAX_RESPONSE_TOKENS = 4_096
    INPUT_POLICY = "background_navigation_prediction_rule_only"
    BACKGROUND_LIST_FIELDS = (
        "objects",
        "processes",
        "conditions",
        "symbols_and_units",
        "missing_information",
        "inactive_context",
    )

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
        max_response_tokens: int | None = None,
        json_retries: int | None = None,
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
        self.max_response_tokens = min(
            self.HARD_MAX_RESPONSE_TOKENS,
            max(256, int(max_response_tokens or self.MAX_RESPONSE_TOKENS)),
        )
        self.json_retries = min(
            self.MAX_JSON_RETRIES,
            max(0, int(self.MAX_JSON_RETRIES if json_retries is None else json_retries)),
        )
        self._client = client
        self._trace_run_active = False
        self._active_stage = ""
        self._background_analysis = self._empty_background_analysis()
        self.last_trace: Dict[str, Any] = {}
        self.last_partial_result: Dict[str, Any] = {}
        self._reset_trace()

    @classmethod
    def _empty_background_analysis(cls) -> Dict[str, Any]:
        return {
            "task_focus": "",
            "objects": [],
            "processes": [],
            "conditions": [],
            "target_quantity": "",
            "symbols_and_units": [],
            "missing_information": [],
            "inactive_context": [],
        }

    def _reset_trace(self) -> None:
        self._background_analysis = self._empty_background_analysis()
        self.last_partial_result = {}
        self.last_trace = {
            "input_policy": self.INPUT_POLICY,
            "model_config": {
                "model": self.model,
                "temperature": self.temperature,
                "thinking_disabled": self._thinking_disabled(),
                "max_response_tokens": self.max_response_tokens,
                "json_attempts": self.json_retries + 1,
                "empty_navigation_recheck": self.json_retries > 0,
            },
            "background_analysis": copy.deepcopy(self._background_analysis),
            "stages": {},
            "partial_result": {},
            "terminal_stage": "",
            "empty_reason": "",
            "status": "running",
        }
        self._active_stage = ""

    def _stage_trace(self, stage: str) -> Dict[str, Any]:
        stages = self.last_trace.setdefault("stages", {})
        trace = stages.setdefault(
            stage,
            {
                "candidate_count": 0,
                "candidates": [],
                "not_selected": [],
                "model_response": None,
                "model_responses": [],
                "api_attempts": [],
                "accepted": [],
                "rejected": [],
                "empty_reason": "",
            },
        )
        return trace

    def _begin_stage(self, stage: str, *, candidate_count: int = 0, reset: bool = False) -> Dict[str, Any]:
        self._active_stage = stage
        if reset and stage in self.last_trace.get("stages", {}):
            del self.last_trace["stages"][stage]
        trace = self._stage_trace(stage)
        trace["candidate_count"] = int(trace.get("candidate_count") or 0) + max(0, int(candidate_count))
        return trace

    def _trace_accept(self, stage: str, item: Dict[str, Any]) -> None:
        self._stage_trace(stage)["accepted"].append(copy.deepcopy(item))

    def _trace_candidates(self, stage: str, items: Iterable[Dict[str, Any]]) -> None:
        trace = self._stage_trace(stage)
        trace["candidates"].extend(copy.deepcopy(list(items)))

    def _trace_not_selected(
        self,
        stage: str,
        items: Iterable[Dict[str, Any]],
        *,
        returned_ids: Iterable[str],
        id_key: str,
    ) -> None:
        returned = {norm_text(item) for item in returned_ids if norm_text(item)}
        trace = self._stage_trace(stage)
        for item in items:
            candidate_id = norm_text(item.get(id_key) or "")
            if candidate_id and candidate_id not in returned:
                trace["not_selected"].append(
                    {
                        **copy.deepcopy(item),
                        "reason": "not_returned_by_model",
                    }
                )

    def _trace_reject(self, stage: str, item: Any, reason: str, **context: Any) -> None:
        record: Dict[str, Any] = {"item": copy.deepcopy(item), "reason": norm_text(reason)}
        record.update({key: copy.deepcopy(value) for key, value in context.items()})
        self._stage_trace(stage)["rejected"].append(record)

    def _set_empty_reason(self, stage: str, reason: str) -> None:
        self._stage_trace(stage)["empty_reason"] = norm_text(reason)

    @staticmethod
    def _strict_bool(value: Any) -> Optional[bool]:
        return value if isinstance(value, bool) else None

    @staticmethod
    def _thinking_disabled() -> bool:
        return norm_text(os.getenv("OPENAI_DISABLE_THINKING") or "") in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _safe_score(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            score = float(value)
        except (TypeError, ValueError):
            return None
        if score != score or score in {float("inf"), float("-inf")}:
            return None
        return max(0.0, min(score, 1.0))

    @classmethod
    def _validate_background_analysis_contract(cls, response: Dict[str, Any]) -> None:
        analysis = response.get("background_analysis")
        if not isinstance(analysis, dict):
            raise RuntimeError("'background_analysis' must be a JSON object.")
        task_focus = analysis.get("task_focus")
        if not isinstance(task_focus, str) or not norm_text(task_focus):
            raise RuntimeError("'background_analysis.task_focus' must be a non-empty string.")

    @classmethod
    def _validate_selection_contract(
        cls,
        response: Dict[str, Any],
        *,
        list_key: str,
        bool_key: str,
        resolve_id: Callable[[Dict[str, Any]], str],
    ) -> None:
        items = response.get(list_key)
        if not isinstance(items, list):
            raise RuntimeError(f"'{list_key}' must be a JSON array.")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise RuntimeError(f"'{list_key}[{index}]' must be a JSON object.")
            if not norm_text(resolve_id(item)):
                raise RuntimeError(f"'{list_key}[{index}]' references an unknown candidate.")
            if cls._strict_bool(item.get(bool_key)) is None:
                raise RuntimeError(f"'{list_key}[{index}].{bool_key}' must be a JSON boolean.")
            raw_score = item.get("score")
            if isinstance(raw_score, bool):
                raise RuntimeError(f"'{list_key}[{index}].score' must be a number from 0 to 1.")
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = float("nan")
            if score != score or score in {float("inf"), float("-inf")} or not 0.0 <= score <= 1.0:
                raise RuntimeError(f"'{list_key}[{index}].score' must be a number from 0 to 1.")

    @staticmethod
    def _canonical_key(value: Any) -> str:
        text = norm_text(value or "").casefold().replace("&", " and ")
        return "".join(char for char in text if char.isalnum())

    @classmethod
    def _normalize_background_analysis(cls, value: Any) -> Dict[str, Any]:
        result = cls._empty_background_analysis()
        if not isinstance(value, dict):
            return result
        result["task_focus"] = norm_text(value.get("task_focus") or "")
        result["target_quantity"] = norm_text(value.get("target_quantity") or "")
        for field in cls.BACKGROUND_LIST_FIELDS:
            raw_items = value.get(field)
            if isinstance(raw_items, dict):
                raw_items = [f"{key}: {item}" for key, item in raw_items.items()]
            elif not isinstance(raw_items, list):
                raw_items = [raw_items] if norm_text(raw_items or "") else []
            result[field] = ordered_unique(
                [norm_text(item) for item in raw_items if norm_text(item)]
            )[:24]
        return result

    def _background_prompt_fields(
        self,
        sample: Dict[str, Any],
        background_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        return {
            "problem_background": self._problem_background(sample),
            "background_analysis": copy.deepcopy(
                background_analysis if background_analysis is not None else self._background_analysis
            ),
        }

    @staticmethod
    def _compact_judgment(item: Dict[str, Any]) -> Dict[str, Any]:
        excluded = {"topic_obj", "cluster_obj", "rule_obj", "topic_rules", "rule_groups"}
        return {key: copy.deepcopy(value) for key, value in item.items() if key not in excluded}

    def _update_partial_result(
        self,
        *,
        domain_result: Dict[str, Any],
        topic_result: Dict[str, Any] | None = None,
        cluster_result: Dict[str, Any] | None = None,
        rule_result: Dict[str, Any] | None = None,
    ) -> None:
        topics = topic_result or {"topic_judgments": [], "selected_topics": []}
        clusters = cluster_result or {"cluster_judgments": [], "selected_clusters": []}
        rules = rule_result or {"rule_judgments": [], "selected_rules": []}
        self.last_partial_result = {
            "input_policy": self.INPUT_POLICY,
            "background_analysis": copy.deepcopy(self._background_analysis),
            "domain_judgments": copy.deepcopy(domain_result.get("domain_judgments") or []),
            "topic_judgments": copy.deepcopy(topics.get("topic_judgments") or []),
            "cluster_judgments": copy.deepcopy(clusters.get("cluster_judgments") or []),
            "rule_judgments": copy.deepcopy(rules.get("rule_judgments") or []),
            "selected_domains": copy.deepcopy(domain_result.get("selected_domains") or []),
            "selected_topics": copy.deepcopy(topics.get("selected_topics") or []),
            "selected_clusters": copy.deepcopy(clusters.get("selected_clusters") or []),
            "selected_rules": copy.deepcopy(rules.get("selected_rules") or []),
        }
        self._sync_trace_partial_result()

    def _sync_trace_partial_result(self) -> None:
        compact: Dict[str, Any] = {}
        for key, value in self.last_partial_result.items():
            if key in {"topic_judgments", "selected_topics", "cluster_judgments", "selected_clusters", "rule_judgments", "selected_rules"}:
                compact[key] = [self._compact_judgment(item) for item in value if isinstance(item, dict)]
            else:
                compact[key] = copy.deepcopy(value)
        self.last_trace["partial_result"] = compact

    def _checkpoint_partial_items(self, stage: str, items: Iterable[Dict[str, Any]]) -> None:
        """Keep completed calls visible if a later call in the same stage fails."""
        keys = {
            "cluster": ("cluster_judgments", "selected_clusters"),
            "rule": ("rule_judgments", "selected_rules"),
        }
        if stage not in keys:
            return
        if not self.last_partial_result:
            self.last_partial_result = {
                "input_policy": self.INPUT_POLICY,
                "background_analysis": copy.deepcopy(self._background_analysis),
                "domain_judgments": [],
                "topic_judgments": [],
                "cluster_judgments": [],
                "rule_judgments": [],
                "selected_domains": [],
                "selected_topics": [],
                "selected_clusters": [],
                "selected_rules": [],
            }
        judgments_key, selected_key = keys[stage]
        checkpoint = copy.deepcopy(list(items))
        self.last_partial_result[judgments_key] = checkpoint
        self.last_partial_result[selected_key] = copy.deepcopy(checkpoint)
        self._sync_trace_partial_result()

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

    def _chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        list_key: str | None = None,
        response_validator: Callable[[Dict[str, Any]], None] | None = None,
        contract_hint: str = "",
        retry_empty_selection: bool = False,
        selection_bool_key: str | None = None,
    ) -> Dict[str, Any]:
        client = self._get_client()
        request: Dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_response_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self._thinking_disabled():
            request["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        stage = self._active_stage or "chat_json"
        stage_trace = self._stage_trace(stage)
        request_index = int(stage_trace.get("chat_call_count") or 0) + 1
        stage_trace["chat_call_count"] = request_index
        last_error: Exception | None = None
        active_request = request
        empty_selection_rechecked = False
        for attempt in range(1, self.json_retries + 2):
            try:
                response = client.chat.completions.create(**active_request)
            except Exception as exc:
                stage_trace["api_attempts"].append(
                    {
                        "request_index": request_index,
                        "attempt": attempt,
                        "raw": "",
                        "finish_reason": None,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise
            choices = response.choices if getattr(response, "choices", None) else []
            choice = choices[0] if choices else None
            raw_content = getattr(getattr(choice, "message", None), "content", "") if choice else ""
            raw = "" if raw_content is None else str(raw_content)
            finish_reason = getattr(choice, "finish_reason", None) if choice else None
            attempt_trace: Dict[str, Any] = {
                "request_index": request_index,
                "attempt": attempt,
                "raw": raw,
                "finish_reason": finish_reason,
            }
            try:
                parsed = self._extract_json_object(raw, list_key=list_key)
                if list_key and not isinstance(parsed.get(list_key), list):
                    raise RuntimeError(
                        "Semantic matcher JSON must contain a top-level "
                        f"'{list_key}' array."
                    )
                if response_validator is not None:
                    response_validator(parsed)
                selection_items = parsed.get(list_key) if list_key else None
                no_positive_selection = not selection_items
                if selection_bool_key and isinstance(selection_items, list):
                    no_positive_selection = not any(
                        isinstance(item, dict)
                        and self._strict_bool(item.get(selection_bool_key)) is True
                        for item in selection_items
                    )
                if (
                    retry_empty_selection
                    and list_key
                    and no_positive_selection
                    and not empty_selection_rechecked
                    and attempt < self.json_retries + 1
                ):
                    empty_selection_rechecked = True
                    raise RuntimeError(
                        f"'{list_key}' has no positive selection; reconsider the supplied candidates once "
                        "before confirming no match."
                    )
            except RuntimeError as exc:
                last_error = exc
                attempt_trace["error"] = str(exc)
                stage_trace["api_attempts"].append(attempt_trace)
                expected_shape = (
                    f"a top-level JSON object whose '{list_key}' field is an array"
                    if list_key
                    else "a top-level JSON object"
                )
                if contract_hint:
                    expected_shape = f"{expected_shape}; {contract_hint}"
                active_request = dict(request)
                active_request["messages"] = [
                    request["messages"][0],
                    {
                        "role": "user",
                        "content": (
                            f"{user_prompt}\n\nYour previous response could not be accepted: {exc} Return "
                            f"{expected_shape} only. Do not return a scalar number, prose, "
                            "analysis, or Markdown fencing."
                        ),
                    },
                ]
                continue
            stage_trace["api_attempts"].append(attempt_trace)
            stage_trace["model_response"] = copy.deepcopy(parsed)
            stage_trace["model_responses"].append(copy.deepcopy(parsed))
            return parsed
        assert last_error is not None
        raise RuntimeError(
            f"Invalid semantic JSON after {self.json_retries + 1} attempts: {last_error}"
        ) from last_error

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

    @classmethod
    def _build_domain_candidates(cls, catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for domain in catalog.get("domains", []) or []:
            if not isinstance(domain, dict):
                continue
            topics = [topic for topic in (domain.get("topics") or []) if isinstance(topic, dict)]
            topic_names = [norm_text(topic.get("name") or "") for topic in topics if norm_text(topic.get("name") or "")]
            domain_name = norm_text(domain.get("name") or "Unknown")
            domain_id = norm_text(domain.get("id") or domain.get("domain_id") or "")
            if not domain_id:
                domain_id = cls._canonical_key(domain_name) or "unknown_domain"
            out.append(
                {
                    "domain_id": domain_id,
                    "domain": domain_name,
                    "summary": norm_text(domain.get("summary") or ""),
                    "topic_count": len(topics),
                    "sample_topics": topic_names[:5],
                }
            )
        return out

    @classmethod
    def _compact_retrieval_hints(cls, topic: Dict[str, Any]) -> Dict[str, Any]:
        hints = topic.get("retrieval_hints") if isinstance(topic.get("retrieval_hints"), dict) else {}
        limits = {
            "scene_keywords": (5, 70),
            "topic_keywords": (6, 40),
            "required_symbols": (6, 25),
            "llm_discriminative_terms": (3, 70),
            "llm_problem_phrases": (1, 120),
        }
        compact: Dict[str, Any] = {}
        for field, (item_limit, char_limit) in limits.items():
            values = hints.get(field) if isinstance(hints.get(field), list) else []
            clipped = ordered_unique(
                [cls._clip_prompt_text(value, char_limit) for value in values if norm_text(value)]
            )[:item_limit]
            if clipped:
                compact[field] = clipped
        return compact

    @classmethod
    def _topic_cluster_previews(cls, topic: Dict[str, Any]) -> List[Dict[str, Any]]:
        previews: List[Dict[str, Any]] = []
        for cluster in topic.get("scenario_clusters", []) or []:
            if not isinstance(cluster, dict):
                continue
            activation_conditions = []
            for group in cluster.get("rule_groups", []) or []:
                if not isinstance(group, dict):
                    continue
                condition = norm_text(group.get("activation_condition") or "")
                if condition:
                    activation_conditions.append(cls._clip_prompt_text(condition, 120))
            previews.append(
                {
                    "cluster_id": norm_text(cluster.get("id") or cluster.get("cluster_id") or ""),
                    "name": cls._clip_prompt_text(cluster.get("name") or "", 80),
                    "summary": cls._clip_prompt_text(cluster.get("summary") or "", 160),
                    "activation_conditions": ordered_unique(activation_conditions)[:1],
                }
            )
            if len(previews) >= 2:
                break
        return previews

    @classmethod
    def _build_topic_candidates(cls, catalog: Dict[str, Any], domains: Iterable[str]) -> List[Dict[str, Any]]:
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
                topic_name = norm_text(topic.get("name") or "Unknown")
                topic_id = norm_text(topic.get("id") or topic.get("topic_id") or "")
                if not topic_id:
                    domain_key = cls._canonical_key(domain_name) or "unknown_domain"
                    topic_key = cls._canonical_key(topic_name) or "unknown_topic"
                    topic_id = f"{domain_key}.{topic_key}"
                out.append(
                    {
                        "domain": domain_name,
                        "topic_id": topic_id,
                        "topic": topic_name,
                        "summary": norm_text(topic.get("summary") or ""),
                        "rule_count": len(topic.get("rules") or []),
                        "retrieval_hints": cls._compact_retrieval_hints(topic),
                        "cluster_previews": cls._topic_cluster_previews(topic),
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
        if not self._trace_run_active:
            self._reset_trace()
        domain_candidates = self._build_domain_candidates(catalog)
        self._begin_stage("domain", candidate_count=len(domain_candidates), reset=True)
        domain_trace_candidates = [
            {"domain_id": item["domain_id"], "domain": item["domain"]}
            for item in domain_candidates
        ]
        self._trace_candidates("domain", domain_trace_candidates)
        if not domain_candidates:
            self._set_empty_reason("domain", "no_domain_candidates")
            return {
                "background_analysis": copy.deepcopy(self._background_analysis),
                "domain_judgments": [],
                "selected_domains": [],
            }
        prompt_payload = {
            **self._background_prompt_fields(sample, self._empty_background_analysis()),
            "candidate_domains": domain_candidates,
            "max_selected_domains": self.MAX_SELECTED_DOMAINS,
            "output_schema": {
                "background_analysis": self._empty_background_analysis(),
                "domains": [
                    {
                        "domain_id": "stable candidate domain_id",
                        "relevant": True,
                        "score": 0.0,
                        "reason": "short reason",
                    }
                ]
            },
        }
        candidates_by_id = {item["domain_id"]: item for item in domain_candidates}
        candidates_by_name = {
            self._canonical_key(item["domain"]): item for item in domain_candidates
        }

        def resolve_domain_id(item: Dict[str, Any]) -> str:
            domain_id = norm_text(item.get("domain_id") or "")
            if domain_id in candidates_by_id:
                return domain_id
            candidate = candidates_by_name.get(
                self._canonical_key(item.get("domain") or item.get("canonical_name") or "")
            )
            return norm_text(candidate.get("domain_id") or "") if candidate else ""

        def validate_domain_response(payload: Dict[str, Any]) -> None:
            self._validate_background_analysis_contract(payload)
            self._validate_selection_contract(
                payload,
                list_key="domains",
                bool_key="relevant",
                resolve_id=resolve_domain_id,
            )

        response = self._chat_json(
            system_prompt=(
                "You are a physics rule navigator. First extract a structured background_analysis from the raw "
                "question and context: task_focus, objects, processes, conditions, target_quantity, "
                "symbols_and_units, missing_information, and inactive_context. Then select the minimum set of "
                "physics domains whose laws apply to that stated physical system. The task focus and active "
                "conditions must dominate historical, illustrative, or otherwise inactive context. Do not infer "
                "problem facts from any student solution. Reject adjacent domains that match only by vocabulary. "
                "If the background is incomplete, explicitly list what is missing, but still select at least one "
                "broad domain when the active task is recognizably a physics problem. Copy the stable domain_id "
                "exactly; do not rewrite it. Return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            list_key="domains",
            response_validator=validate_domain_response,
            retry_empty_selection=True,
            selection_bool_key="relevant",
            contract_hint=(
                "background_analysis must be an object with a non-empty task_focus; "
                "every domain item must use a known candidate id, a JSON boolean relevant, and a numeric score"
            ),
        )
        self._background_analysis = self._normalize_background_analysis(response.get("background_analysis"))
        self.last_trace["background_analysis"] = copy.deepcopy(self._background_analysis)
        response_items = response.get("domains")
        if not isinstance(response_items, list):
            self._trace_reject("domain", response_items, "domains_must_be_a_list")
            response_items = []
        judgments: List[Dict[str, Any]] = []
        returned_domain_ids = [
            resolve_domain_id(item) for item in response_items if isinstance(item, dict)
        ]
        self._trace_not_selected(
            "domain",
            domain_trace_candidates,
            returned_ids=returned_domain_ids,
            id_key="domain_id",
        )
        for item in response_items:
            if not isinstance(item, dict):
                self._trace_reject("domain", item, "domain_item_must_be_an_object")
                continue
            domain_id = norm_text(item.get("domain_id") or "")
            candidate = candidates_by_id.get(domain_id) if domain_id else None
            if candidate is None:
                candidate = candidates_by_name.get(
                    self._canonical_key(item.get("domain") or item.get("canonical_name") or "")
                )
            if candidate is None:
                self._trace_reject("domain", item, "unknown_domain")
                continue
            relevant = self._strict_bool(item.get("relevant"))
            if relevant is None:
                self._trace_reject("domain", item, "relevant_must_be_json_boolean")
                continue
            if not relevant:
                self._trace_reject("domain", item, "model_marked_irrelevant")
                continue
            score = self._safe_score(item.get("score"))
            if score is None:
                self._trace_reject("domain", item, "invalid_score")
                continue
            judgments.append(
                {
                    "domain_id": candidate["domain_id"],
                    "domain": candidate["domain"],
                    "relevant": True,
                    "score": score,
                    "reason": norm_text(item.get("reason") or ""),
                }
            )
        best_by_domain: Dict[str, Dict[str, Any]] = {}
        for judgment in judgments:
            current = best_by_domain.get(judgment["domain_id"])
            if current is None or float(judgment["score"]) > float(current["score"]):
                if current is not None:
                    self._trace_reject("domain", self._compact_judgment(current), "duplicate_lower_score")
                best_by_domain[judgment["domain_id"]] = judgment
            else:
                self._trace_reject("domain", self._compact_judgment(judgment), "duplicate_lower_score")
        judgments = list(best_by_domain.values())
        judgments.sort(key=lambda item: (-float(item["score"]), item["domain"]))
        for omitted in judgments[self.MAX_SELECTED_DOMAINS :]:
            self._trace_reject("domain", self._compact_judgment(omitted), "selection_limit")
        judgments = judgments[: self.MAX_SELECTED_DOMAINS]
        for judgment in judgments:
            self._trace_accept("domain", self._compact_judgment(judgment))
        if not judgments:
            reason = "model_selected_no_domains" if not response_items else "all_domain_items_rejected"
            self._set_empty_reason("domain", reason)
        return {
            "background_analysis": copy.deepcopy(self._background_analysis),
            "domain_judgments": judgments,
            "selected_domains": [item["domain"] for item in judgments],
        }

    def select_topics_semantically(
        self,
        sample: Dict[str, Any],
        catalog: Dict[str, Any],
        domains: Iterable[str],
        background_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        selected_domain_names = [norm_text(item) for item in domains if norm_text(item)]
        if not selected_domain_names:
            self._begin_stage("topic", reset=True)
            self._set_empty_reason("topic", "no_selected_domains")
            return {"topic_judgments": [], "selected_topics": []}
        topic_candidates = self._build_topic_candidates(catalog, selected_domain_names)
        self._begin_stage("topic", candidate_count=len(topic_candidates), reset=True)
        topic_trace_candidates = [
            {
                "domain": item["domain"],
                "topic_id": item["topic_id"],
                "topic": item["topic"],
            }
            for item in topic_candidates
        ]
        self._trace_candidates("topic", topic_trace_candidates)
        if not topic_candidates:
            self._set_empty_reason("topic", "no_topic_candidates")
            return {"topic_judgments": [], "selected_topics": []}
        prompt_payload = {
            **self._background_prompt_fields(sample, background_analysis),
            "max_selected_topics": self.MAX_SELECTED_TOPICS,
            "candidate_topics": [
                {
                    "domain": item["domain"],
                    "topic_id": item["topic_id"],
                    "canonical_name": item["topic"],
                    "summary": item["summary"],
                    "retrieval_hints": item["retrieval_hints"],
                    "cluster_previews": item["cluster_previews"],
                    "rule_count": item["rule_count"],
                }
                for item in topic_candidates
            ],
            "output_schema": {
                "topics": [
                    {
                        "topic_id": "stable candidate topic_id",
                        "relevant": True,
                        "score": 0.0,
                        "reason": "short reason",
                    }
                ]
            },
        }
        candidate_by_id = {item["topic_id"]: item for item in topic_candidates}
        candidates_by_name: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for candidate in topic_candidates:
            key = (self._canonical_key(candidate["domain"]), self._canonical_key(candidate["topic"]))
            candidates_by_name.setdefault(key, []).append(candidate)

        def resolve_topic_id(item: Dict[str, Any]) -> str:
            topic_id = norm_text(item.get("topic_id") or "")
            if topic_id:
                return topic_id if topic_id in candidate_by_id else ""
            domain_key = self._canonical_key(item.get("domain") or "")
            topic_key = self._canonical_key(item.get("topic") or item.get("canonical_name") or "")
            matches = candidates_by_name.get((domain_key, topic_key), [])
            if not matches and topic_key:
                matches = [
                    value
                    for (candidate_domain, candidate_topic), values in candidates_by_name.items()
                    if candidate_topic == topic_key
                    for value in values
                ]
            return matches[0]["topic_id"] if len(matches) == 1 else ""

        response = self._chat_json(
            system_prompt=(
                "You are a physics rule navigator. Inside the selected domains, use only the problem background to "
                "choose the minimum set of topics that govern the physical mechanism. Prefer 1-2 topics. Reject "
                "neighboring concepts, prerequisite knowledge, downstream consequences, and shared-symbol matches. "
                "Do not infer problem facts from a student solution. Treat topic summaries as hard semantic "
                "boundaries. Use retrieval hints and cluster previews only to disambiguate those boundaries. Copy "
                "the stable topic_id exactly; do not rewrite it. If the background is incomplete, be conservative. "
                "Missing a figure or a fine-grained condition is not by itself a reason to omit the broad topic; "
                "record the missing information and select the closest supported physical mechanism. Return JSON only."
            ),
            user_prompt=json.dumps(prompt_payload, ensure_ascii=False, indent=2),
            list_key="topics",
            response_validator=lambda payload: self._validate_selection_contract(
                payload,
                list_key="topics",
                bool_key="relevant",
                resolve_id=resolve_topic_id,
            ),
            retry_empty_selection=True,
            selection_bool_key="relevant",
            contract_hint=(
                "every topic item must use a known candidate id, a JSON boolean relevant, and a numeric score"
            ),
        )
        response_items = response.get("topics")
        if not isinstance(response_items, list):
            self._trace_reject("topic", response_items, "topics_must_be_a_list")
            response_items = []
        returned_topic_ids = [
            resolve_topic_id(item) for item in response_items if isinstance(item, dict)
        ]
        self._trace_not_selected(
            "topic",
            topic_trace_candidates,
            returned_ids=returned_topic_ids,
            id_key="topic_id",
        )
        judgments: List[Dict[str, Any]] = []
        for item in response_items:
            if not isinstance(item, dict):
                self._trace_reject("topic", item, "topic_item_must_be_an_object")
                continue
            topic_id = norm_text(item.get("topic_id") or "")
            candidate = candidate_by_id.get(topic_id) if topic_id else None
            if topic_id and candidate is None:
                self._trace_reject("topic", item, "unknown_topic_id")
                continue
            if candidate is None:
                domain_key = self._canonical_key(item.get("domain") or "")
                topic_key = self._canonical_key(item.get("topic") or item.get("canonical_name") or "")
                matches = candidates_by_name.get((domain_key, topic_key), [])
                if not matches and topic_key:
                    matches = [
                        value for (candidate_domain, candidate_topic), values in candidates_by_name.items()
                        if candidate_topic == topic_key
                        for value in values
                    ]
                if len(matches) != 1:
                    self._trace_reject("topic", item, "unknown_or_ambiguous_canonical_topic")
                    continue
                candidate = matches[0]
                topic_id = candidate["topic_id"]
            relevant = self._strict_bool(item.get("relevant"))
            if relevant is None:
                self._trace_reject("topic", item, "relevant_must_be_json_boolean")
                continue
            if not relevant:
                self._trace_reject("topic", item, "model_marked_irrelevant")
                continue
            score = self._safe_score(item.get("score"))
            if score is None:
                self._trace_reject("topic", item, "invalid_score")
                continue
            judgments.append(
                {
                    "domain": candidate["domain"],
                    "topic_id": topic_id,
                    "topic": candidate["topic"],
                    "relevant": True,
                    "score": score,
                    "reason": norm_text(item.get("reason") or ""),
                    "topic_obj": candidate["topic_obj"],
                }
            )
        best_by_topic: Dict[str, Dict[str, Any]] = {}
        for judgment in judgments:
            key = judgment["topic_id"]
            current = best_by_topic.get(key)
            if current is None or float(judgment["score"]) > float(current["score"]):
                if current is not None:
                    self._trace_reject("topic", self._compact_judgment(current), "duplicate_lower_score")
                best_by_topic[key] = judgment
            else:
                self._trace_reject("topic", self._compact_judgment(judgment), "duplicate_lower_score")
        judgments = list(best_by_topic.values())
        judgments.sort(key=lambda item: (-float(item["score"]), item["topic_id"]))
        for omitted in judgments[self.MAX_SELECTED_TOPICS :]:
            self._trace_reject("topic", self._compact_judgment(omitted), "selection_limit")
        judgments = judgments[: self.MAX_SELECTED_TOPICS]
        for judgment in judgments:
            self._trace_accept("topic", self._compact_judgment(judgment))
        if not judgments:
            reason = "model_selected_no_topics" if not response_items else "all_topic_items_rejected"
            self._set_empty_reason("topic", reason)
        return {"topic_judgments": judgments, "selected_topics": judgments}

    def select_clusters_semantically(
        self,
        sample: Dict[str, Any],
        selected_topics: Iterable[Dict[str, Any]],
        background_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self._begin_stage("cluster", reset=True)
        all_judgments: List[Dict[str, Any]] = []
        selected_topic_list = list(selected_topics)
        for topic_match in selected_topic_list:
            cluster_candidates = self._build_cluster_candidates(topic_match)
            if not cluster_candidates:
                continue
            self._begin_stage("cluster", candidate_count=len(cluster_candidates))
            cluster_trace_candidates = [
                {
                    "domain": item["domain"],
                    "topic_id": topic_match.get("topic_id") or "",
                    "topic": item["topic"],
                    "cluster_id": item["cluster_id"],
                    "cluster": item["cluster"],
                }
                for item in cluster_candidates
            ]
            self._trace_candidates("cluster", cluster_trace_candidates)
            prompt_payload = {
                **self._background_prompt_fields(sample, background_analysis),
                "max_selected_clusters": self.MAX_SELECTED_CLUSTERS,
                "domain": topic_match["domain"],
                "topic_id": topic_match.get("topic_id") or "",
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
            cluster_index = {item["cluster_id"]: item for item in cluster_candidates}

            def resolve_cluster_id(item: Dict[str, Any]) -> str:
                cluster_id = norm_text(item.get("cluster_id") or "")
                return cluster_id if cluster_id in cluster_index else ""

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
                response_validator=lambda payload: self._validate_selection_contract(
                    payload,
                    list_key="clusters",
                    bool_key="relevant",
                    resolve_id=resolve_cluster_id,
                ),
                retry_empty_selection=True,
                selection_bool_key="relevant",
                contract_hint=(
                    "every cluster item must use a known candidate id, a JSON boolean relevant, and a numeric score"
                ),
            )
            topic_judgments: List[Dict[str, Any]] = []
            response_items = response.get("clusters")
            if not isinstance(response_items, list):
                self._trace_reject(
                    "cluster",
                    response_items,
                    "clusters_must_be_a_list",
                    topic_id=topic_match.get("topic_id") or "",
                )
                response_items = []
            returned_cluster_ids = [
                resolve_cluster_id(item) for item in response_items if isinstance(item, dict)
            ]
            self._trace_not_selected(
                "cluster",
                cluster_trace_candidates,
                returned_ids=returned_cluster_ids,
                id_key="cluster_id",
            )
            for item in response_items:
                if not isinstance(item, dict):
                    self._trace_reject("cluster", item, "cluster_item_must_be_an_object")
                    continue
                cluster_id = norm_text(item.get("cluster_id") or "")
                if cluster_id not in cluster_index:
                    self._trace_reject("cluster", item, "unknown_cluster_id")
                    continue
                relevant = self._strict_bool(item.get("relevant"))
                if relevant is None:
                    self._trace_reject("cluster", item, "relevant_must_be_json_boolean")
                    continue
                if not relevant:
                    self._trace_reject("cluster", item, "model_marked_irrelevant")
                    continue
                score = self._safe_score(item.get("score"))
                if score is None:
                    self._trace_reject("cluster", item, "invalid_score")
                    continue
                candidate = cluster_index[cluster_id]
                topic_judgments.append(
                    {
                        "cluster_id": cluster_id,
                        "cluster": candidate["cluster"],
                        "domain": candidate["domain"],
                        "topic_id": topic_match.get("topic_id") or "",
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
            self._checkpoint_partial_items("cluster", all_judgments)
        best_by_cluster: Dict[tuple[str, str, str], Dict[str, Any]] = {}
        for judgment in all_judgments:
            key = (judgment["domain"], judgment.get("topic_id") or judgment["topic"], judgment["cluster_id"])
            current = best_by_cluster.get(key)
            if current is None or float(judgment["score"]) > float(current["score"]):
                if current is not None:
                    self._trace_reject("cluster", self._compact_judgment(current), "duplicate_lower_score")
                best_by_cluster[key] = judgment
            else:
                self._trace_reject("cluster", self._compact_judgment(judgment), "duplicate_lower_score")
        all_judgments = list(best_by_cluster.values())
        all_judgments.sort(key=lambda item: (-float(item["score"]), item["domain"], item["topic"], item["cluster_id"]))
        topic_order = {
            (item["domain"], item.get("topic_id") or item["topic"]): index
            for index, item in enumerate(selected_topic_list)
        }
        grouped: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for judgment in all_judgments:
            key = (judgment["domain"], judgment.get("topic_id") or judgment["topic"])
            grouped.setdefault(key, []).append(judgment)
        primaries = [
            items[0]
            for _key, items in sorted(
                grouped.items(),
                key=lambda pair: topic_order.get(pair[0], len(topic_order)),
            )
            if items
        ]
        primary_ids = {id(item) for item in primaries}
        extras = [item for item in all_judgments if id(item) not in primary_ids]
        fair_order = primaries + extras
        for omitted in fair_order[self.MAX_SELECTED_CLUSTERS :]:
            self._trace_reject("cluster", self._compact_judgment(omitted), "selection_limit")
        all_judgments = fair_order[: self.MAX_SELECTED_CLUSTERS]
        for judgment in all_judgments:
            self._trace_accept("cluster", self._compact_judgment(judgment))
        stage_trace = self._stage_trace("cluster")
        if not all_judgments:
            reason = (
                "no_cluster_candidates"
                if not stage_trace["candidate_count"]
                else "model_selected_no_clusters_or_all_items_rejected"
            )
            self._set_empty_reason("cluster", reason)
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
        background_analysis: Dict[str, Any] | None,
        context_domain: str,
        context_topic: str,
        topic_obj: Dict[str, Any],
        rule_candidates: List[Dict[str, Any]],
        cluster_id: str = "",
        cluster_name: str = "",
        cluster_description: str = "",
        rule_group_summaries: List[Dict[str, Any]] | None = None,
        candidate_source: str = "topic_fallback",
        checkpoint_callback: Callable[[List[Dict[str, Any]]], None] | None = None,
    ) -> List[Dict[str, Any]]:
        batches = self._batch_rule_candidates(rule_candidates)
        if not batches:
            return []
        self._begin_stage("rule", candidate_count=len(rule_candidates))
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
            batch_rule_index = {candidate["rule_id"]: candidate for candidate, _ in batch}
            rule_trace_candidates = [
                {
                    "batch_index": batch_index,
                    "batch_total": len(batches),
                    "domain": context_domain,
                    "topic": context_topic,
                    "cluster_id": cluster_id,
                    "rule_id": candidate["rule_id"],
                    "title": candidate["title"],
                    "candidate_source": candidate_source,
                }
                for candidate, _ in batch
            ]
            self._trace_candidates("rule", rule_trace_candidates)

            def resolve_rule_id(item: Dict[str, Any]) -> str:
                rule_id = norm_text(item.get("rule_id") or "")
                return rule_id if rule_id in batch_rule_index else ""

            prompt_payload = {
                **self._background_prompt_fields(sample, background_analysis),
                "student_solution": self._student_solution(sample),
                "domain": context_domain,
                "topic": context_topic,
                "candidate_source": candidate_source,
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
                response_validator=lambda payload: self._validate_selection_contract(
                    payload,
                    list_key="rules",
                    bool_key="applicable",
                    resolve_id=resolve_rule_id,
                ),
                contract_hint=(
                    "every rule item must use an id from this batch, a JSON boolean applicable, and a numeric score"
                ),
            )
            response_items = response.get("rules")
            if not isinstance(response_items, list):
                self._trace_reject(
                    "rule",
                    response_items,
                    "rules_must_be_a_list",
                    candidate_source=candidate_source,
                    batch_index=batch_index,
                )
                response_items = []
            returned_rule_ids = [
                resolve_rule_id(item) for item in response_items if isinstance(item, dict)
            ]
            self._trace_not_selected(
                "rule",
                rule_trace_candidates,
                returned_ids=returned_rule_ids,
                id_key="rule_id",
            )
            for item in response_items:
                if not isinstance(item, dict):
                    self._trace_reject("rule", item, "rule_item_must_be_an_object", candidate_source=candidate_source)
                    continue
                rule_id = norm_text(item.get("rule_id") or "")
                if rule_id not in batch_rule_index:
                    self._trace_reject("rule", item, "unknown_rule_id", candidate_source=candidate_source)
                    continue
                applicable = self._strict_bool(item.get("applicable"))
                if applicable is None:
                    self._trace_reject(
                        "rule", item, "applicable_must_be_json_boolean", candidate_source=candidate_source
                    )
                    continue
                if not applicable:
                    self._trace_reject("rule", item, "model_marked_not_applicable", candidate_source=candidate_source)
                    continue
                score = self._safe_score(item.get("score"))
                if score is None:
                    self._trace_reject("rule", item, "invalid_score", candidate_source=candidate_source)
                    continue
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
                        "candidate_source": candidate_source,
                        "rule_obj": candidate["rule_obj"],
                    }
                )
            if checkpoint_callback is not None:
                checkpoint_callback(judgments)
        best_by_rule: Dict[str, Dict[str, Any]] = {}
        for judgment in judgments:
            current = best_by_rule.get(judgment["rule_id"])
            if current is None or float(judgment["score"]) > float(current["score"]):
                if current is not None:
                    self._trace_reject("rule", self._compact_judgment(current), "duplicate_lower_score")
                best_by_rule[judgment["rule_id"]] = judgment
            else:
                self._trace_reject("rule", self._compact_judgment(judgment), "duplicate_lower_score")
        merged = list(best_by_rule.values())
        merged.sort(key=lambda item: (-float(item["score"]), item["rule_id"]))
        return merged

    def select_rules_semantically(
        self,
        sample: Dict[str, Any],
        selected_topics: Iterable[Dict[str, Any]],
        selected_clusters: Iterable[Dict[str, Any]] | None = None,
        background_analysis: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        self._begin_stage("rule", reset=True)
        all_judgments: List[Dict[str, Any]] = []
        selected_topic_list = list(selected_topics)
        cluster_list = list(selected_clusters or [])
        topics_with_rule_hits: set[tuple[str, str]] = set()
        attempted_rule_ids: Dict[tuple[str, str], set[str]] = {}

        for item in cluster_list:
            topic_key = (item["domain"], item["topic"])
            topic_match = next(
                (topic for topic in selected_topic_list if (topic["domain"], topic["topic"]) == topic_key),
                item,
            )
            all_topic_candidates = self._build_rule_candidates(topic_match)
            allowed_ids = {norm_text(rule_id) for rule_id in item.get("rule_ids", []) or [] if norm_text(rule_id)}
            rule_candidates = [candidate for candidate in all_topic_candidates if candidate["rule_id"] in allowed_ids]
            if not rule_candidates:
                continue
            attempted_rule_ids.setdefault(topic_key, set()).update(candidate["rule_id"] for candidate in rule_candidates)
            topic_obj = item.get("topic_obj") if isinstance(item.get("topic_obj"), dict) else {}
            cluster_judgments = self._select_rules_for_context(
                sample=sample,
                background_analysis=background_analysis,
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
                candidate_source="selected_cluster",
                checkpoint_callback=lambda current: self._checkpoint_partial_items(
                    "rule", [*all_judgments, *current]
                ),
            )
            if cluster_judgments:
                topics_with_rule_hits.add(topic_key)
            all_judgments.extend(cluster_judgments)

        # Only clusterless topics use a topic-wide rule pass. When scenario clusters
        # exist, their semantic boundary remains hard; broadening to hundreds of
        # unrelated topic rules would increase both false positives and API cost.
        # General reasoning remains a last-resort fallback below.
        general_clusters: Dict[tuple[str, str], Dict[str, Any]] = {}
        for topic_match in selected_topic_list:
            topic_key = (topic_match["domain"], topic_match["topic"])
            topic_cluster_candidates = self._build_cluster_candidates(topic_match)
            general_cluster = next(
                (
                    cluster
                    for cluster in topic_cluster_candidates
                    if cluster["cluster_id"] == "general_reasoning"
                ),
                None,
            )
            if general_cluster:
                general_clusters[topic_key] = general_cluster
            if topic_key in topics_with_rule_hits:
                continue
            if any(cluster["cluster_id"] != "general_reasoning" for cluster in topic_cluster_candidates):
                continue
            already_attempted = attempted_rule_ids.setdefault(topic_key, set())
            general_rule_ids = set((general_cluster or {}).get("rule_ids") or [])
            rule_candidates = [
                candidate
                for candidate in self._build_rule_candidates(topic_match)
                if candidate["rule_id"] not in already_attempted
                and candidate["rule_id"] not in general_rule_ids
            ]
            if not rule_candidates:
                continue
            already_attempted.update(candidate["rule_id"] for candidate in rule_candidates)
            topic_judgments = self._select_rules_for_context(
                sample=sample,
                background_analysis=background_analysis,
                context_domain=topic_match["domain"],
                context_topic=topic_match["topic"],
                topic_obj=topic_match.get("topic_obj") if isinstance(topic_match.get("topic_obj"), dict) else {},
                rule_candidates=rule_candidates,
                candidate_source="topic_fallback",
                checkpoint_callback=lambda current: self._checkpoint_partial_items(
                    "rule", [*all_judgments, *current]
                ),
            )
            if topic_judgments:
                topics_with_rule_hits.add(topic_key)
            all_judgments.extend(topic_judgments)

        for topic_match in selected_topic_list:
            topic_key = (topic_match["domain"], topic_match["topic"])
            if topic_key in topics_with_rule_hits:
                continue
            general_cluster = general_clusters.get(topic_key)
            if not general_cluster:
                continue
            already_attempted = attempted_rule_ids.setdefault(topic_key, set())
            general_rule_ids = set(general_cluster.get("rule_ids") or [])
            rule_candidates = [
                candidate
                for candidate in self._build_rule_candidates(topic_match)
                if candidate["rule_id"] in general_rule_ids
                and candidate["rule_id"] not in already_attempted
            ]
            if not rule_candidates:
                continue
            already_attempted.update(candidate["rule_id"] for candidate in rule_candidates)
            general_judgments = self._select_rules_for_context(
                sample=sample,
                background_analysis=background_analysis,
                context_domain=topic_match["domain"],
                context_topic=topic_match["topic"],
                topic_obj=topic_match.get("topic_obj") if isinstance(topic_match.get("topic_obj"), dict) else {},
                rule_candidates=rule_candidates,
                cluster_id="general_reasoning",
                cluster_name=general_cluster["cluster"],
                cluster_description=general_cluster["summary"],
                candidate_source="general_reasoning_fallback",
                checkpoint_callback=lambda current: self._checkpoint_partial_items(
                    "rule", [*all_judgments, *current]
                ),
            )
            if general_judgments:
                topics_with_rule_hits.add(topic_key)
            all_judgments.extend(general_judgments)

        best_by_rule: Dict[str, Dict[str, Any]] = {}
        for judgment in all_judgments:
            rule_id = judgment["rule_id"]
            current = best_by_rule.get(rule_id)
            if current is None or float(judgment["score"]) > float(current["score"]):
                if current is not None:
                    self._trace_reject("rule", self._compact_judgment(current), "duplicate_lower_score")
                best_by_rule[rule_id] = judgment
            else:
                self._trace_reject("rule", self._compact_judgment(judgment), "duplicate_lower_score")
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
                self._trace_reject("rule", self._compact_judgment(judgment), "clusterless_topic_limit")
                continue
            clusterless_kept[topic_key] = kept + 1
            capped.append(judgment)
        topic_order = {
            (item["domain"], item["topic"]): index
            for index, item in enumerate(selected_topic_list)
        }
        grouped_rules: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for judgment in capped:
            grouped_rules.setdefault((judgment["domain"], judgment["topic"]), []).append(judgment)
        primary_rules = [
            items[0]
            for key, items in sorted(
                grouped_rules.items(),
                key=lambda pair: topic_order.get(pair[0], len(topic_order)),
            )
            if items
        ]
        primary_ids = {id(item) for item in primary_rules}
        fair_order = primary_rules + [item for item in capped if id(item) not in primary_ids]
        for omitted in fair_order[self.max_selected_rules :]:
            self._trace_reject("rule", self._compact_judgment(omitted), "global_selection_limit")
        globally_capped = fair_order[: self.max_selected_rules]
        for judgment in globally_capped:
            self._trace_accept("rule", self._compact_judgment(judgment))
        if not globally_capped:
            reason = (
                "no_rule_candidates"
                if not self._stage_trace("rule")["candidate_count"]
                else "model_selected_no_rules_or_all_items_rejected"
            )
            self._set_empty_reason("rule", reason)
        return {"rule_judgments": globally_capped, "selected_rules": globally_capped}

    def _tree_result(
        self,
        *,
        domain_result: Dict[str, Any],
        topic_result: Dict[str, Any] | None = None,
        cluster_result: Dict[str, Any] | None = None,
        rule_result: Dict[str, Any] | None = None,
        terminal_stage: str,
        empty_reason: str = "",
    ) -> Dict[str, Any]:
        self._update_partial_result(
            domain_result=domain_result,
            topic_result=topic_result,
            cluster_result=cluster_result,
            rule_result=rule_result,
        )
        self.last_trace["background_analysis"] = copy.deepcopy(self._background_analysis)
        self.last_trace["terminal_stage"] = norm_text(terminal_stage)
        self.last_trace["empty_reason"] = norm_text(empty_reason)
        self.last_trace["status"] = "empty" if empty_reason else "complete"
        result = copy.deepcopy(self.last_partial_result)
        result["terminal_stage"] = norm_text(terminal_stage)
        result["empty_reason"] = norm_text(empty_reason)
        result["navigation_trace"] = copy.deepcopy(self.last_trace)
        return result

    def _raise_stage_error(self, stage: str, exc: Exception) -> None:
        self.last_trace["terminal_stage"] = stage
        self.last_trace["empty_reason"] = "selection_error"
        self.last_trace["status"] = "failed"
        self.last_trace["error"] = f"{type(exc).__name__}: {exc}"
        raise SemanticSelectionError(
            stage,
            exc,
            trace=self.last_trace,
            partial_result=self.last_partial_result,
        ) from exc

    def select_tree_semantically(self, sample: Dict[str, Any], catalog: Dict[str, Any]) -> Dict[str, Any]:
        self._reset_trace()
        self._trace_run_active = True
        domain_result: Dict[str, Any] = {"domain_judgments": [], "selected_domains": []}
        topic_result: Dict[str, Any] = {"topic_judgments": [], "selected_topics": []}
        cluster_result: Dict[str, Any] = {"cluster_judgments": [], "selected_clusters": []}
        try:
            try:
                domain_result = self.select_domains_semantically(sample, catalog)
                self._update_partial_result(domain_result=domain_result)
            except Exception as exc:
                self._raise_stage_error("domain", exc)
            if not domain_result["selected_domains"]:
                empty_reason = self._stage_trace("domain")["empty_reason"] or "no_selected_domains"
                return self._tree_result(
                    domain_result=domain_result,
                    terminal_stage="domain",
                    empty_reason=empty_reason,
                )
            try:
                topic_result = self.select_topics_semantically(
                    sample,
                    catalog,
                    domain_result["selected_domains"],
                    domain_result.get("background_analysis"),
                )
                self._update_partial_result(domain_result=domain_result, topic_result=topic_result)
            except Exception as exc:
                self._raise_stage_error("topic", exc)
            if not topic_result["selected_topics"]:
                empty_reason = self._stage_trace("topic")["empty_reason"] or "no_selected_topics"
                return self._tree_result(
                    domain_result=domain_result,
                    topic_result=topic_result,
                    terminal_stage="topic",
                    empty_reason=empty_reason,
                )
            try:
                cluster_result = self.select_clusters_semantically(
                    sample,
                    topic_result["selected_topics"],
                    domain_result.get("background_analysis"),
                )
                self._update_partial_result(
                    domain_result=domain_result,
                    topic_result=topic_result,
                    cluster_result=cluster_result,
                )
            except Exception as exc:
                self._raise_stage_error("cluster", exc)
            try:
                rule_result = self.select_rules_semantically(
                    sample,
                    topic_result["selected_topics"],
                    cluster_result["selected_clusters"],
                    domain_result.get("background_analysis"),
                )
            except Exception as exc:
                self._raise_stage_error("rule", exc)
            empty_reason = ""
            if not rule_result["selected_rules"]:
                empty_reason = self._stage_trace("rule")["empty_reason"] or "no_selected_rules"
            return self._tree_result(
                domain_result=domain_result,
                topic_result=topic_result,
                cluster_result=cluster_result,
                rule_result=rule_result,
                terminal_stage="rule",
                empty_reason=empty_reason,
            )
        finally:
            self._trace_run_active = False
            self._active_stage = ""
