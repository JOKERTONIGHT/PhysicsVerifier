from __future__ import annotations

import unittest

from core.unified_semantic_matcher import UnifiedSemanticMatcher


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def create(self, **_: object) -> _FakeResponse:
        if not self._responses:
            raise AssertionError("No fake responses left.")
        return _FakeResponse(self._responses.pop(0))


class _FakeChat:
    def __init__(self, responses: list[str]) -> None:
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self.chat = _FakeChat(responses)


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

    def test_select_tree_semantically_returns_structured_results(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(
                [
                    '{"domains":[{"domain":"Modern Physics","relevant":true,"score":0.91,"reason":"relativity setup"}]}',
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

    def test_rule_selection_falls_back_to_topic_when_cluster_path_returns_no_rule(self) -> None:
        matcher = UnifiedSemanticMatcher(
            model="fake-model",
            client=_FakeClient(
                [
                    '{"domains":[{"domain":"Modern Physics","relevant":true,"score":0.91,"reason":"relativity setup"}]}',
                    '{"topics":[{"domain":"Modern Physics","topic":"Special Relativity (Time Dilation, Length Contraction)","relevant":true,"score":0.89,"reason":"moving rod under pinhole observation"}]}',
                    '{"clusters":[]}',
                    '{"rules":[{"rule_id":"exp_pinhole","applicable":true,"score":0.93,"reason":"topic fallback still identifies the pinhole rule"}]}',
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
        self.assertEqual(len(result["selected_rules"]), 1)
        self.assertEqual(result["selected_rules"][0]["rule_id"], "exp_pinhole")

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
                    '{"domains":[{"domain":"Electromagnetism","relevant":true,"score":0.95,"reason":"circuit problem"}]}',
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
                    '{"domains":[{"domain":"Mechanics","relevant":true,"score":1.0,"reason":"mechanics"}]}',
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
