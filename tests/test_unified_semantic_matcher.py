from __future__ import annotations

import copy
import json
import unittest

from core.unified_semantic_matcher import SemanticSelectionError, UnifiedSemanticMatcher


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.message = _FakeMessage(content)
        self.finish_reason = finish_reason


class _FakeResponse:
    def __init__(self, content: str, finish_reason: str = "stop") -> None:
        self.choices = [_FakeChoice(content, finish_reason)]


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> _FakeResponse:
        if not self._responses:
            raise AssertionError("No fake responses left.")
        self.requests.append(request)
        return _FakeResponse(self._responses.pop(0))


class _FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = _FakeChat(responses)

    @property
    def requests(self) -> list[dict[str, object]]:
        return self.chat.completions.requests


def _catalog() -> dict:
    return {
        "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2", "schema_profile": "semantic_navigation_tree_minimal"},
        "domains": [
            {
                "id": "mechanics",
                "name": "Mechanics",
                "summary": "Classical mechanics topics.",
                "topics": [
                    {
                        "id": "mechanics.gravitation_and_keplers_laws",
                        "name": "Gravitation and Kepler's Laws",
                        "summary": "Orbital motion and gravity-governed trajectories.",
                        "rules": [
                            {
                                "rule_id": "exp_orbit_decay",
                                "title": "Orbital decay energy accounting",
                                "summary": "Check satellite orbital decay energy accounting.",
                                "trigger": "satellite orbital decay with drag",
                                "check_logic": "track orbital energy loss consistently",
                                "error_type": "logic",
                                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["G", "M", "r"]},
                            }
                        ],
                        "scenario_clusters": [
                            {
                                "id": "orbital_decay_and_orbit_accounting",
                                "name": "Orbital Decay and Orbit Accounting",
                                "summary": "Satellite orbit change and orbital-energy bookkeeping.",
                                "rule_groups": [
                                    {
                                        "id": "orbit_decay_core_checks",
                                        "name": "Orbit Decay Core Checks",
                                        "summary": "Checks orbital decay accounting.",
                                        "activation_condition": "Use for orbital decay scenarios.",
                                        "rule_ids": ["exp_orbit_decay"],
                                    }
                                ],
                                "rule_ids": ["exp_orbit_decay"],
                            }
                        ],
                    }
                ],
            },
            {
                "id": "modern_physics",
                "name": "Modern Physics",
                "summary": "Relativity and quantum topics.",
                "topics": [
                    {
                        "id": "modern_physics.special_relativity_time_dilation_length_contraction",
                        "name": "Special Relativity (Time Dilation, Length Contraction)",
                        "summary": "Relativistic observation and frame effects.",
                        "rules": [
                            {
                                "rule_id": "exp_pinhole",
                                "title": "Pinhole simultaneity",
                                "summary": "Check simultaneity assumptions in pinhole observations.",
                                "trigger": "pinhole camera sees moving rod",
                                "check_logic": "treat exposure as simultaneous in observer frame",
                                "error_type": "logic",
                                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["L", "v"]},
                            }
                        ],
                        "scenario_clusters": [
                            {
                                "id": "observation_and_projection",
                                "name": "Observation and Projection Geometry",
                                "summary": "Observation tasks where camera timing and projection are central.",
                                "rule_groups": [
                                    {
                                        "id": "projection_checks",
                                        "name": "Projection Checks",
                                        "summary": "Checks relativistic observation geometry.",
                                        "activation_condition": "Use for observed length problems.",
                                        "rule_ids": ["exp_pinhole"],
                                    }
                                ],
                                "rule_ids": ["exp_pinhole"],
                            }
                        ],
                    }
                ],
            },
        ],
    }


def _background_analysis(task_focus: str = "Classify the stated physics task.") -> dict:
    return {
        "task_focus": task_focus,
        "objects": [],
        "processes": [],
        "conditions": [],
        "target_quantity": "",
        "symbols_and_units": [],
        "missing_information": [],
        "inactive_context": [],
    }


def _domain_json(domains: list[dict], task_focus: str = "Classify the stated physics task.") -> str:
    return json.dumps(
        {
            "background_analysis": _background_analysis(task_focus),
            "domains": domains,
        }
    )


class UnifiedSemanticMatcherTests(unittest.TestCase):
    def test_chat_json_accepts_fenced_json_object(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(['```json\n{"domains":[{"domain":"Mechanics","relevant":true,"score":0.9,"reason":"motion"}]}\n```']),
        )

        result = matcher._chat_json(system_prompt="system", user_prompt="user")

        self.assertEqual(result["domains"][0]["domain"], "Mechanics")

    def test_chat_json_wraps_array_response_for_expected_selection_key(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(['[{"domain":"Mechanics","topic":"Gravitation and Kepler\'s Laws","relevant":true,"score":0.95,"reason":"orbit"}]']),
        )

        result = matcher._chat_json(system_prompt="system", user_prompt="user", list_key="topics")

        self.assertEqual(result["topics"][0]["topic"], "Gravitation and Kepler's Laws")

    def test_chat_json_retries_invalid_content_and_records_bounded_attempts(self) -> None:
        client = _FakeClient(
            [
                "0.00000000000021",
                "not json",
                '{"domains":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            api_key="must-not-appear-in-trace",
            json_retries=3,
            max_response_tokens=999_999,
        )

        result = matcher._chat_json(system_prompt="system", user_prompt="user", list_key="domains")

        self.assertEqual(result, {"domains": []})
        self.assertEqual(len(client.requests), 3)
        self.assertTrue(all(request["max_tokens"] == matcher.HARD_MAX_RESPONSE_TOKENS for request in client.requests))
        attempts = matcher.last_trace["stages"]["chat_json"]["api_attempts"]
        self.assertEqual([item["raw"] for item in attempts], ["0.00000000000021", "not json", '{"domains":[]}'])
        self.assertEqual([item["finish_reason"] for item in attempts], ["stop", "stop", "stop"])
        self.assertNotIn("must-not-appear-in-trace", json.dumps(matcher.last_trace, ensure_ascii=False))
        retry_messages = client.requests[1]["messages"]
        self.assertIn("top-level JSON object", retry_messages[-1]["content"])
        self.assertIn("'domains' field is an array", retry_messages[-1]["content"])

    def test_chat_json_stops_after_initial_request_plus_three_retries(self) -> None:
        client = _FakeClient(["bad-1", "bad-2", "bad-3", "bad-4", '{"rules":[]}'])
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client, json_retries=99)

        with self.assertRaisesRegex(RuntimeError, "after 4 attempts"):
            matcher._chat_json(system_prompt="system", user_prompt="user", list_key="rules")

        self.assertEqual(len(client.requests), 4)
        self.assertEqual(len(matcher.last_trace["stages"]["chat_json"]["api_attempts"]), 4)

    def test_chat_json_retries_missing_or_non_array_selection_key(self) -> None:
        client = _FakeClient(
            [
                '{"wrong":[]}',
                '{"domains":{}}',
                '{"domains":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client, json_retries=2)

        result = matcher._chat_json(system_prompt="system", user_prompt="user", list_key="domains")

        self.assertEqual(result, {"domains": []})
        self.assertEqual(len(client.requests), 3)
        errors = [
            item.get("error", "")
            for item in matcher.last_trace["stages"]["chat_json"]["api_attempts"]
        ]
        self.assertIn("'domains' array", errors[0])
        self.assertIn("'domains' array", errors[1])

    def test_domain_retries_missing_or_wrong_background_analysis(self) -> None:
        domain_item = {
            "domain": "Mechanics",
            "relevant": True,
            "score": 0.9,
            "reason": "orbital motion",
        }
        valid_analysis = _background_analysis("Classify a satellite-orbit problem.")
        client = _FakeClient(
            [
                json.dumps({"domains": [domain_item]}),
                json.dumps({"background_analysis": [], "domains": [domain_item]}),
                json.dumps({"background_analysis": valid_analysis, "domains": [domain_item]}),
                '{"topics":[]}',
                '{"topics":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client, json_retries=3)

        result = matcher.select_tree_semantically(
            {"question": "A satellite moves in orbit.", "context": "", "prediction": "claim"},
            _catalog(),
        )

        self.assertEqual(result["background_analysis"], valid_analysis)
        attempts = result["navigation_trace"]["stages"]["domain"]["api_attempts"]
        self.assertEqual(len(attempts), 3)
        self.assertIn("background_analysis", attempts[0]["error"])
        self.assertIn("background_analysis", attempts[1]["error"])
        topic_payload = json.loads(client.requests[-2]["messages"][-1]["content"])
        self.assertEqual(topic_payload["background_analysis"], valid_analysis)

    def test_background_analysis_and_topic_id_flow_through_navigation(self) -> None:
        catalog = copy.deepcopy(_catalog())
        topic = catalog["domains"][1]["topics"][0]
        topic["retrieval_hints"] = {
            "scene_keywords": ["pinhole observation", "moving rod"],
            "llm_discriminative_terms": ["observer-frame exposure"],
        }
        analysis = {
            "task_focus": "audit the observed rod length",
            "objects": ["rod", "pinhole camera"],
            "processes": ["relativistic observation"],
            "conditions": ["camera frame is specified"],
            "target_quantity": "observed length",
            "symbols_and_units": ["v: speed"],
            "missing_information": [],
            "inactive_context": ["unrelated apparatus history"],
        }
        client = _FakeClient(
            [
                json.dumps(
                    {
                        "background_analysis": analysis,
                        "domains": [
                            {
                                "domain_id": "modern_physics",
                                "relevant": True,
                                "score": 0.95,
                                "reason": "relativistic observation",
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "topics": [
                            {
                                "topic_id": "modern_physics.special_relativity_time_dilation_length_contraction",
                                "relevant": True,
                                "score": 0.94,
                                "reason": "moving rod observation",
                            }
                        ]
                    }
                ),
                '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":0.9,"reason":"pinhole"}]}',
                '{"rules":[{"rule_id":"exp_pinhole","applicable":true,"score":0.93,"reason":"auditable"}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)
        sample = {
            "question": "RAW_QUESTION: A pinhole camera observes a moving rod.",
            "context": "RAW_CONTEXT: Work in the camera frame.",
            "prediction": "UNTRUSTED_PREDICTION: exposure is simultaneous.",
        }

        result = matcher.select_tree_semantically(sample, catalog)

        self.assertEqual(result["background_analysis"], analysis)
        self.assertEqual(result["domain_judgments"][0]["domain_id"], "modern_physics")
        self.assertEqual(
            result["selected_topics"][0]["topic_id"],
            "modern_physics.special_relativity_time_dilation_length_contraction",
        )
        payloads = [
            json.loads(next(message for message in request["messages"] if message["role"] == "user")["content"])
            for request in client.requests
        ]
        topic_candidate = payloads[1]["candidate_topics"][0]
        self.assertEqual(topic_candidate["topic_id"], result["selected_topics"][0]["topic_id"])
        self.assertIn("retrieval_hints", topic_candidate)
        self.assertIn("cluster_previews", topic_candidate)
        for payload in payloads[1:]:
            self.assertEqual(payload["background_analysis"], analysis)
            self.assertIn(sample["question"], payload["problem_background"])
            self.assertIn(sample["context"], payload["problem_background"])
        self.assertNotIn(sample["prediction"], json.dumps(payloads[:3], ensure_ascii=False))
        self.assertEqual(payloads[3]["student_solution"], sample["prediction"])
        stages = result["navigation_trace"]["stages"]
        self.assertEqual(
            {item["domain_id"] for item in stages["domain"]["candidates"]},
            {"mechanics", "modern_physics"},
        )
        self.assertIn(
            "mechanics",
            {item["domain_id"] for item in stages["domain"]["not_selected"]},
        )
        self.assertEqual(stages["domain"]["not_selected"][0]["reason"], "not_returned_by_model")
        self.assertIn("exp_pinhole", {item["rule_id"] for item in stages["rule"]["candidates"]})
        self.assertEqual(stages["rule"]["candidates"][0]["batch_index"], 1)

    def test_invalid_domain_items_trigger_contract_retry(self) -> None:
        client = _FakeClient(
            [
                json.dumps(
                    {
                        "background_analysis": _background_analysis(),
                        "domains": [
                            {"domain": "Mechanics", "relevant": "false", "score": 1.0, "reason": "bad bool"},
                            {"domain": "Unknown Domain", "relevant": True, "score": 1.0, "reason": "unknown"},
                        ]
                    }
                ),
                _domain_json(
                    [{"domain": "Mechanics", "relevant": True, "score": 0.9, "reason": "valid"}]
                ),
                '{"topics":[]}',
                '{"topics":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {"question": "A satellite moves in orbit.", "context": "", "prediction": ""},
            _catalog(),
        )

        self.assertEqual(result["selected_domains"], ["Mechanics"])
        attempts = result["navigation_trace"]["stages"]["domain"]["api_attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertIn("JSON boolean", attempts[0]["error"])

    def test_out_of_range_selection_score_triggers_retry(self) -> None:
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Mechanics", "relevant": True, "score": 95, "reason": "invalid scale"}]
                ),
                _domain_json(
                    [{"domain": "Mechanics", "relevant": True, "score": 0.95, "reason": "valid scale"}]
                ),
                '{"topics":[]}',
                '{"topics":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {"question": "A mechanics problem.", "context": "", "prediction": "claim"},
            _catalog(),
        )

        attempts = result["navigation_trace"]["stages"]["domain"]["api_attempts"]
        self.assertEqual(len(attempts), 2)
        self.assertIn("0 to 1", attempts[0]["error"])

    def test_stage_failure_exposes_partial_result_and_raw_attempts(self) -> None:
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Modern Physics", "relevant": True, "score": 0.9, "reason": "modern"}]
                ),
                '{"topics":[{"topic_id":"modern_physics.special_relativity_time_dilation_length_contraction","relevant":true,"score":0.9,"reason":"relativity"}]}',
                "0.1",
                "0.2",
                "0.3",
                "0.4",
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        with self.assertRaises(SemanticSelectionError) as caught:
            matcher.select_tree_semantically(
                {"question": "A pinhole camera observes a moving rod.", "context": "", "prediction": "claim"},
                _catalog(),
            )

        error = caught.exception
        self.assertEqual(error.stage, "cluster")
        self.assertEqual(error.partial_result["selected_domains"], ["Modern Physics"])
        self.assertEqual(len(error.partial_result["selected_topics"]), 1)
        self.assertEqual(error.trace["status"], "failed")
        self.assertEqual(error.trace["terminal_stage"], "cluster")
        attempts = error.trace["stages"]["cluster"]["api_attempts"]
        self.assertEqual([item["raw"] for item in attempts], ["0.1", "0.2", "0.3", "0.4"])

    def test_second_topic_cluster_failure_keeps_first_topic_cluster(self) -> None:
        client = _FakeClient(
            [
                _domain_json(
                    [
                        {"domain": "Mechanics", "relevant": True, "score": 1.0, "reason": "orbit"},
                        {"domain": "Modern Physics", "relevant": True, "score": 0.9, "reason": "observation"},
                    ]
                ),
                json.dumps(
                    {
                        "topics": [
                            {
                                "topic_id": "mechanics.gravitation_and_keplers_laws",
                                "relevant": True,
                                "score": 1.0,
                                "reason": "orbit",
                            },
                            {
                                "topic_id": "modern_physics.special_relativity_time_dilation_length_contraction",
                                "relevant": True,
                                "score": 0.9,
                                "reason": "observation",
                            },
                        ]
                    }
                ),
                '{"clusters":[{"cluster_id":"orbital_decay_and_orbit_accounting","relevant":true,"score":0.95,"reason":"orbit decay"}]}',
                "bad-1",
                "bad-2",
                "bad-3",
                "bad-4",
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        with self.assertRaises(SemanticSelectionError) as caught:
            matcher.select_tree_semantically(
                {"question": "Compare an orbit and an observation setup.", "context": "", "prediction": "claim"},
                _catalog(),
            )

        self.assertEqual(caught.exception.stage, "cluster")
        self.assertEqual(
            [item["cluster_id"] for item in caught.exception.partial_result["selected_clusters"]],
            ["orbital_decay_and_orbit_accounting"],
        )

    def test_later_rule_batch_failure_keeps_earlier_rule_hit(self) -> None:
        rules = [
            {
                "rule_id": f"r{index}",
                "title": f"Rule {index}",
                "summary": "A candidate check.",
                "trigger": "stated setup",
                "check_logic": "audit the claim",
                "error_type": "logic",
                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
            }
            for index in range(2)
        ]
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "id": "test_domain",
                    "name": "Test Domain",
                    "summary": "A test domain.",
                    "topics": [
                        {
                            "id": "test_domain.test_topic",
                            "name": "Test Topic",
                            "summary": "A clusterless test topic.",
                            "rules": rules,
                            "scenario_clusters": [],
                        }
                    ],
                }
            ],
        }
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain_id": "test_domain", "relevant": True, "score": 1.0, "reason": "test"}]
                ),
                '{"topics":[{"topic_id":"test_domain.test_topic","relevant":true,"score":1.0,"reason":"test"}]}',
                '{"rules":[{"rule_id":"r0","applicable":true,"score":0.9,"reason":"first hit"}]}',
                "bad-1",
                "bad-2",
                "bad-3",
                "bad-4",
            ]
        )
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            rule_candidate_batch_size=1,
        )

        with self.assertRaises(SemanticSelectionError) as caught:
            matcher.select_tree_semantically(
                {"question": "A test setup.", "context": "", "prediction": "A claim."},
                catalog,
            )

        self.assertEqual(caught.exception.stage, "rule")
        self.assertEqual(
            [item["rule_id"] for item in caught.exception.partial_result["selected_rules"]],
            ["r0"],
        )

    @staticmethod
    def _catalog_with_general_reasoning() -> dict:
        catalog = copy.deepcopy(_catalog())
        topic = catalog["domains"][1]["topics"][0]
        topic["rules"].append(
            {
                "rule_id": "exp_general_reasoning",
                "title": "General consistency check",
                "summary": "Audit a general logical consistency condition.",
                "trigger": "a derived claim",
                "check_logic": "compare the claim with stated conditions",
                "error_type": "logic",
                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
            }
        )
        topic["scenario_clusters"].append(
            {
                "id": "general_reasoning",
                "name": "General Reasoning",
                "summary": "Cross-scenario consistency checks.",
                "rule_groups": [],
                "rule_ids": ["exp_general_reasoning"],
            }
        )
        return catalog

    def test_general_reasoning_is_not_evaluated_after_specific_rule_hit(self) -> None:
        catalog = self._catalog_with_general_reasoning()
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Modern Physics", "relevant": True, "score": 1.0, "reason": "modern"}]
                ),
                '{"topics":[{"topic_id":"modern_physics.special_relativity_time_dilation_length_contraction","relevant":true,"score":1.0,"reason":"relativity"}]}',
                '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":1.0,"reason":"specific"}]}',
                '{"rules":[{"rule_id":"exp_pinhole","applicable":true,"score":0.95,"reason":"specific audit"}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {"question": "A pinhole camera observes a moving rod.", "context": "", "prediction": "A claim."},
            catalog,
        )

        self.assertEqual(
            [item["rule_id"] for item in result["selected_rules"]],
            ["exp_pinhole"],
        )
        sources = {item["candidate_source"] for item in result["navigation_trace"]["stages"]["rule"]["accepted"]}
        self.assertEqual(sources, {"selected_cluster"})
        self.assertEqual(len(client.requests), 4)

    def test_general_reasoning_is_used_after_specific_cluster_returns_no_rule(self) -> None:
        catalog = self._catalog_with_general_reasoning()
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Modern Physics", "relevant": True, "score": 1.0, "reason": "modern"}]
                ),
                '{"topics":[{"topic_id":"modern_physics.special_relativity_time_dilation_length_contraction","relevant":true,"score":1.0,"reason":"relativity"}]}',
                '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":1.0,"reason":"specific"}]}',
                '{"rules":[]}',
                '{"rules":[{"rule_id":"exp_general_reasoning","applicable":true,"score":0.9,"reason":"general audit"}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {"question": "A pinhole camera observes a moving rod.", "context": "", "prediction": "A claim."},
            catalog,
        )

        self.assertEqual([item["rule_id"] for item in result["selected_rules"]], ["exp_general_reasoning"])
        self.assertEqual(result["selected_rules"][0]["candidate_source"], "general_reasoning_fallback")
        self.assertEqual(len(client.requests), 5)

    def test_select_tree_semantically_returns_structured_results(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(
                [
                    _domain_json(
                        [{"domain": "Modern Physics", "relevant": True, "score": 0.91, "reason": "relativity setup"}]
                    ),
                    '{"topics":[{"domain":"Modern Physics","topic":"Special Relativity (Time Dilation, Length Contraction)","relevant":true,"score":0.89,"reason":"moving rod under pinhole observation"}]}',
                    '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":0.87,"reason":"camera observation cluster"}]}',
                    '{"rules":[{"rule_id":"exp_pinhole","applicable":true,"score":0.93,"reason":"matches pinhole moving rod scenario"}]}',
                ]
            ),
        )
        sample = {
            "id": "29185",
            "question": "A pinhole camera observes a rod moving with velocity v.",
            "prediction": "Treat the exposure as simultaneous in the observer frame.",
            "answer": "",
        }
        result = matcher.select_tree_semantically(sample, _catalog())
        self.assertEqual(result["selected_domains"], ["Modern Physics"])
        self.assertEqual(len(result["selected_topics"]), 1)
        self.assertEqual(result["selected_topics"][0]["topic"], "Special Relativity (Time Dilation, Length Contraction)")
        self.assertEqual(len(result["selected_clusters"]), 1)
        self.assertEqual(result["selected_clusters"][0]["cluster_id"], "observation_and_projection")
        self.assertEqual(len(result["selected_rules"]), 1)
        self.assertEqual(result["selected_rules"][0]["rule_id"], "exp_pinhole")

    def test_navigation_uses_background_and_only_rule_selection_receives_prediction(self) -> None:
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Modern Physics", "relevant": True, "score": 0.91, "reason": "relativity setup"}]
                ),
                '{"topics":[{"domain":"Modern Physics","topic":"Special Relativity (Time Dilation, Length Contraction)","relevant":true,"score":0.89,"reason":"moving rod under pinhole observation"}]}',
                '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":0.87,"reason":"camera observation cluster"}]}',
                '{"rules":[{"rule_id":"exp_pinhole","applicable":true,"score":0.93,"reason":"matches the student claim"}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)
        sample = {
            "id": "input_role_boundary",
            "question": "QUESTION_ONLY_SENTINEL: A pinhole camera observes a moving rod.",
            "context": "CONTEXT_ONLY_SENTINEL: The camera frame is specified.",
            "prediction": "PREDICTION_ONLY_SENTINEL: Treat the exposure as simultaneous.",
            "answer": "",
        }

        result = matcher.select_tree_semantically(sample, _catalog())

        self.assertEqual(len(client.requests), 4)
        payloads = []
        for request in client.requests:
            messages = request["messages"]
            self.assertIsInstance(messages, list)
            user_message = next(message for message in messages if message["role"] == "user")
            payloads.append(json.loads(user_message["content"]))

        for payload in payloads[:3]:
            self.assertNotIn("sample", payload)
            self.assertNotIn("student_solution", payload)
            self.assertIn(sample["question"], payload["problem_background"])
            self.assertIn(sample["context"], payload["problem_background"])
            self.assertNotIn(sample["prediction"], json.dumps(payload, ensure_ascii=False))

        rule_payload = payloads[3]
        self.assertNotIn("sample", rule_payload)
        self.assertIn(sample["question"], rule_payload["problem_background"])
        self.assertIn(sample["context"], rule_payload["problem_background"])
        self.assertNotIn(sample["prediction"], rule_payload["problem_background"])
        self.assertEqual(rule_payload["student_solution"], sample["prediction"])
        self.assertEqual(result["input_policy"], "background_navigation_prediction_rule_only")

    def test_empty_domain_selection_short_circuits_the_tree(self) -> None:
        client = _FakeClient(
            [
                _domain_json([], "Identify an ambiguous physical task."),
                _domain_json([], "Identify an ambiguous physical task."),
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {
                "id": "no_domain",
                "question": "An intentionally ambiguous problem statement.",
                "context": "No usable physical setup is provided.",
                "prediction": "This must not be used to reopen the full catalog.",
            },
            _catalog(),
        )

        self.assertEqual(len(client.requests), 2)
        self.assertEqual(result["selected_domains"], [])
        self.assertEqual(result["selected_topics"], [])
        self.assertEqual(result["selected_clusters"], [])
        self.assertEqual(result["selected_rules"], [])

    def test_empty_topic_selection_short_circuits_before_cluster_and_rule(self) -> None:
        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Modern Physics", "relevant": True, "score": 0.9, "reason": "possible modern setup"}]
                ),
                '{"topics":[]}',
                '{"topics":[]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {
                "id": "no_topic",
                "question": "A modern-physics prompt with no matching catalog topic.",
                "context": "",
                "prediction": "This must not trigger cluster or rule expansion.",
            },
            _catalog(),
        )

        self.assertEqual(len(client.requests), 3)
        self.assertEqual(result["selected_domains"], ["Modern Physics"])
        self.assertEqual(result["selected_topics"], [])
        self.assertEqual(result["selected_clusters"], [])
        self.assertEqual(result["selected_rules"], [])

    def test_empty_domain_and_topic_are_rechecked_once_before_acceptance(self) -> None:
        client = _FakeClient(
            [
                "not-json-domain-response",
                _domain_json([], "Audit a relativistic observation."),
                _domain_json(
                    [
                        {
                            "domain": "Modern Physics",
                            "relevant": True,
                            "score": 0.95,
                            "reason": "relativistic observation",
                        }
                    ],
                    "Audit a relativistic observation.",
                ),
                '{"topics":[{"topic_id":"modern_physics.special_relativity_time_dilation_length_contraction","relevant":false,"score":0.2,"reason":"first pass unsure"}]}',
                '{"topics":[{"topic_id":"modern_physics.special_relativity_time_dilation_length_contraction","relevant":true,"score":0.94,"reason":"moving rod"}]}',
                '{"clusters":[]}',
                '{"clusters":[{"cluster_id":"observation_and_projection","relevant":true,"score":0.93,"reason":"camera"}]}',
                '{"rules":[{"rule_id":"exp_pinhole","applicable":true,"score":0.92,"reason":"auditable"}]}',
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {
                "question": "A pinhole camera observes a moving rod.",
                "context": "",
                "prediction": "Treat the exposure as simultaneous.",
            },
            _catalog(),
        )

        self.assertEqual([item["rule_id"] for item in result["selected_rules"]], ["exp_pinhole"])
        self.assertEqual(len(result["navigation_trace"]["stages"]["domain"]["api_attempts"]), 3)
        self.assertEqual(len(result["navigation_trace"]["stages"]["topic"]["api_attempts"]), 2)
        self.assertEqual(len(result["navigation_trace"]["stages"]["cluster"]["api_attempts"]), 2)
        self.assertIn("reconsider", result["navigation_trace"]["stages"]["topic"]["api_attempts"][0]["error"])

    def test_domain_topic_and_cluster_outputs_obey_hard_limits(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "id": f"d{domain_index}",
                    "name": f"Domain {domain_index}",
                    "summary": f"Domain {domain_index} summary.",
                    "topics": [
                        {
                            "id": f"d{domain_index}.t{topic_index}",
                            "name": f"Topic {domain_index}-{topic_index}",
                            "summary": f"Topic {domain_index}-{topic_index} summary.",
                            "rules": [],
                            "scenario_clusters": [
                                {
                                    "id": f"cluster_{domain_index}_{topic_index}_{cluster_index}",
                                    "name": f"Cluster {domain_index}-{topic_index}-{cluster_index}",
                                    "summary": "A candidate scenario.",
                                    "rule_groups": [],
                                    "rule_ids": [],
                                }
                                for cluster_index in range(2)
                            ],
                        }
                        for topic_index in range(2)
                    ],
                }
                for domain_index in range(3)
            ],
        }
        domain_response = {
            "background_analysis": _background_analysis("Classify a broad multi-domain task."),
            "domains": [
                {
                    "domain": f"Domain {index}",
                    "relevant": True,
                    "score": 1.0 - index * 0.1,
                    "reason": "candidate",
                }
                for index in range(3)
            ]
        }
        topic_response = {
            "topics": [
                {
                    "domain": f"Domain {domain_index}",
                    "topic": f"Topic {domain_index}-{topic_index}",
                    "relevant": True,
                    "score": 1.0 - (domain_index * 2 + topic_index) * 0.05,
                    "reason": "candidate",
                }
                for domain_index in range(2)
                for topic_index in range(2)
            ]
        }
        selected_topic_keys = [(0, 0), (0, 1), (1, 0)]
        cluster_responses = [
            {
                "clusters": [
                    {
                        "cluster_id": f"cluster_{domain_index}_{topic_index}_{cluster_index}",
                        "relevant": True,
                        "score": 1.0 - cluster_index * 0.1,
                        "reason": "candidate",
                    }
                    for cluster_index in range(2)
                ]
            }
            for domain_index, topic_index in selected_topic_keys
        ]
        client = _FakeClient(
            [
                json.dumps(domain_response),
                json.dumps(topic_response),
                *[json.dumps(response) for response in cluster_responses],
            ]
        )
        matcher = UnifiedSemanticMatcher(model="fake-model", client=client)

        result = matcher.select_tree_semantically(
            {"question": "A broad multi-domain setup.", "context": "", "prediction": "A proposed solution."},
            catalog,
        )

        self.assertEqual(len(result["selected_domains"]), matcher.MAX_SELECTED_DOMAINS)
        self.assertEqual(len(result["selected_topics"]), matcher.MAX_SELECTED_TOPICS)
        self.assertEqual(len(result["selected_clusters"]), matcher.MAX_SELECTED_CLUSTERS)
        self.assertEqual(len(client.requests), 5)

    def test_large_rule_candidates_are_split_by_character_budget(self) -> None:
        rules = [
            {
                "rule_id": f"batch_rule_{index}",
                "title": f"Batch rule {index}",
                "summary": f"Candidate {index}: " + ("long semantic description " * 80),
                "trigger": "A repeated physical trigger.",
                "check_logic": "Check the corresponding physical claim.",
                "error_type": "logic",
                "preconditions": ["The physical setup activates this candidate."],
                "violation_signatures": ["The proposed solution violates this candidate."],
                "negative_conditions": [],
                "evidence_requirements": ["Quote the relevant solution step."],
                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
            }
            for index in range(5)
        ]
        topic = {
            "id": "batch.topic",
            "name": "Batch Topic",
            "summary": "A topic with many verbose candidate rules.",
            "rules": rules,
            "scenario_clusters": [],
        }
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "id": "batch",
                    "name": "Batch Domain",
                    "summary": "Batching test domain.",
                    "topics": [topic],
                }
            ],
        }
        batch_chars = 2_600
        probe = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient([]),
            rule_candidate_batch_size=100,
            rule_candidate_batch_chars=batch_chars,
        )
        candidates = probe._build_rule_candidates(
            {"domain": "Batch Domain", "topic": "Batch Topic", "topic_obj": topic}
        )
        expected_batches = probe._batch_rule_candidates(candidates)
        self.assertGreater(len(expected_batches), 1)

        client = _FakeClient(
            [
                _domain_json(
                    [{"domain": "Batch Domain", "relevant": True, "score": 1.0, "reason": "batch domain"}]
                ),
                '{"topics":[{"domain":"Batch Domain","topic":"Batch Topic","relevant":true,"score":1.0,"reason":"batch topic"}]}',
                *['{"rules":[]}' for _ in expected_batches],
            ]
        )
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=client,
            rule_candidate_batch_size=100,
            rule_candidate_batch_chars=batch_chars,
        )
        sample = {
            "question": "QUESTION_BATCH_SENTINEL",
            "context": "CONTEXT_BATCH_SENTINEL",
            "prediction": "PREDICTION_BATCH_SENTINEL",
        }

        result = matcher.select_tree_semantically(sample, catalog)

        rule_requests = client.requests[2:]
        self.assertEqual(len(rule_requests), len(expected_batches))
        seen_rule_ids = []
        for request in rule_requests:
            messages = request["messages"]
            user_message = next(message for message in messages if message["role"] == "user")
            payload = json.loads(user_message["content"])
            self.assertIn(sample["question"], payload["problem_background"])
            self.assertIn(sample["context"], payload["problem_background"])
            self.assertNotIn(sample["prediction"], payload["problem_background"])
            self.assertEqual(payload["student_solution"], sample["prediction"])
            self.assertLessEqual(
                len(json.dumps(payload["candidate_rules"], ensure_ascii=False, separators=(",", ":"))),
                batch_chars,
            )
            seen_rule_ids.extend(candidate["rule_id"] for candidate in payload["candidate_rules"])
        self.assertEqual(seen_rule_ids, [rule["rule_id"] for rule in rules])
        self.assertEqual(result["selected_rules"], [])

    def test_rule_selection_does_not_broaden_past_existing_cluster_boundaries(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(
                [
                    _domain_json(
                        [{"domain": "Modern Physics", "relevant": True, "score": 0.91, "reason": "relativity setup"}]
                    ),
                    '{"topics":[{"domain":"Modern Physics","topic":"Special Relativity (Time Dilation, Length Contraction)","relevant":true,"score":0.89,"reason":"moving rod under pinhole observation"}]}',
                    '{"clusters":[]}',
                    '{"clusters":[]}',
                ]
            ),
        )
        sample = {
            "id": "29185",
            "question": "A pinhole camera observes a rod moving with velocity v.",
            "prediction": "Treat the exposure as simultaneous in the observer frame.",
            "answer": "",
        }
        result = matcher.select_tree_semantically(sample, _catalog())
        self.assertEqual(result["selected_clusters"], [])
        self.assertEqual(result["selected_rules"], [])
        self.assertEqual(len(matcher._client.requests), 4)

    def test_clusterless_topic_rules_are_capped(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2", "schema_profile": "semantic_navigation_tree_minimal"},
            "domains": [
                {
                    "id": "electromagnetism",
                    "name": "Electromagnetism",
                    "summary": "Electric and magnetic systems.",
                    "topics": [
                        {
                            "id": "electromagnetism.dc_circuits_and_kirchhoffs_laws",
                            "name": "DC Circuits and Kirchhoff's Laws",
                            "summary": "Static circuit solving with loop and node constraints.",
                            "rules": [
                                {
                                    "rule_id": f"exp_dc_{index}",
                                    "title": f"DC rule {index}",
                                    "summary": f"Check DC loop and node consistency {index}.",
                                    "trigger": "Kirchhoff branch current",
                                    "check_logic": f"Loop and node consistency {index}",
                                    "error_type": "logic",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["I", "V"]},
                                }
                                for index in range(3)
                            ],
                            "scenario_clusters": [],
                        }
                    ],
                }
            ],
        }
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(
                [
                    _domain_json(
                        [{"domain": "Electromagnetism", "relevant": True, "score": 0.95, "reason": "circuit problem"}]
                    ),
                    '{"topics":[{"domain":"Electromagnetism","topic":"DC Circuits and Kirchhoff\'s Laws","relevant":true,"score":0.94,"reason":"static circuit solving"}]}',
                    '{"rules":[{"rule_id":"exp_dc_0","applicable":true,"score":0.95,"reason":"strongly applicable"},{"rule_id":"exp_dc_1","applicable":true,"score":0.9,"reason":"also applicable"},{"rule_id":"exp_dc_2","applicable":true,"score":0.85,"reason":"weaker but still relevant"}]}',
                ]
            ),
        )
        sample = {
            "id": "dc_clusterless_cap",
            "question": "Use Kirchhoff laws to solve branch currents in a DC circuit.",
            "prediction": "Write loop and node equations.",
            "answer": "",
        }
        result = matcher.select_tree_semantically(sample, catalog)
        self.assertEqual(result["selected_clusters"], [])
        self.assertEqual([item["rule_id"] for item in result["selected_rules"]], ["exp_dc_0", "exp_dc_1"])

    def test_selected_rules_are_globally_capped_across_multiple_clusters(self) -> None:
        topic_names = ["Topic A", "Topic B", "Topic C"]
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2", "schema_profile": "semantic_navigation_tree_minimal"},
            "domains": [
                {
                    "id": "mechanics",
                    "name": "Mechanics",
                    "summary": "Mechanics domain.",
                    "topics": [
                        {
                            "id": f"mechanics.topic_{topic_index}",
                            "name": topic_name,
                            "summary": f"{topic_name} summary.",
                            "rules": [
                                {
                                    "rule_id": f"r{topic_index}_{rule_index}",
                                    "title": f"Rule {topic_index}-{rule_index}",
                                    "summary": f"Rule {topic_index}-{rule_index} summary.",
                                    "trigger": "shared trigger",
                                    "check_logic": "shared check",
                                    "error_type": "logic",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
                                }
                                for rule_index in range(3)
                            ],
                            "scenario_clusters": [
                                {
                                    "id": f"cluster_{topic_index}",
                                    "name": f"Cluster {topic_index}",
                                    "summary": f"Cluster {topic_index} summary.",
                                    "rule_groups": [
                                        {
                                            "id": f"group_{topic_index}",
                                            "name": f"Group {topic_index}",
                                            "summary": f"Group {topic_index} summary.",
                                            "activation_condition": "Use for shared trigger.",
                                            "rule_ids": [f"r{topic_index}_{rule_index}" for rule_index in range(3)],
                                        }
                                    ],
                                    "rule_ids": [f"r{topic_index}_{rule_index}" for rule_index in range(3)],
                                }
                            ],
                        }
                        for topic_index, topic_name in enumerate(topic_names)
                    ],
                }
            ],
        }
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(
                [
                    _domain_json(
                        [{"domain": "Mechanics", "relevant": True, "score": 1.0, "reason": "mechanics"}]
                    ),
                    '{"topics":[{"domain":"Mechanics","topic":"Topic A","relevant":true,"score":1.0,"reason":"a"},{"domain":"Mechanics","topic":"Topic B","relevant":true,"score":1.0,"reason":"b"},{"domain":"Mechanics","topic":"Topic C","relevant":true,"score":0.9,"reason":"c"}]}',
                    '{"clusters":[{"cluster_id":"cluster_0","relevant":true,"score":1.0,"reason":"a"}]}',
                    '{"clusters":[{"cluster_id":"cluster_1","relevant":true,"score":1.0,"reason":"b"}]}',
                    '{"clusters":[{"cluster_id":"cluster_2","relevant":true,"score":0.9,"reason":"c"}]}',
                    '{"rules":[{"rule_id":"r0_0","applicable":true,"score":1.0,"reason":"best"},{"rule_id":"r0_1","applicable":true,"score":0.95,"reason":"also"},{"rule_id":"r0_2","applicable":true,"score":0.9,"reason":"weaker"}]}',
                    '{"rules":[{"rule_id":"r1_0","applicable":true,"score":0.98,"reason":"best"},{"rule_id":"r1_1","applicable":true,"score":0.94,"reason":"also"},{"rule_id":"r1_2","applicable":true,"score":0.89,"reason":"weaker"}]}',
                    '{"rules":[{"rule_id":"r2_0","applicable":true,"score":0.97,"reason":"best"},{"rule_id":"r2_1","applicable":true,"score":0.93,"reason":"also"},{"rule_id":"r2_2","applicable":true,"score":0.88,"reason":"weaker"}]}',
                ]
            ),
        )

        result = matcher.select_tree_semantically(
            {"id": "wide", "question": "A multi-mechanism mechanics problem.", "prediction": "Solution.", "answer": ""},
            catalog,
        )

        self.assertEqual(len(result["selected_rules"]), 5)
        self.assertEqual(
            [item["rule_id"] for item in result["selected_rules"]],
            ["r0_0", "r1_0", "r2_0", "r0_1", "r1_1"],
        )


if __name__ == "__main__":
    unittest.main()
