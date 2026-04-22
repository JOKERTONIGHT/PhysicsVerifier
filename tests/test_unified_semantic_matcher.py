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
        "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
        "domains": [
            {
                "name": "Mechanics",
                "topics": [
                    {
                        "name": "Gravitation and Kepler's Laws",
                        "description": "Orbital motion and gravity-governed trajectories.",
                        "includes": ["orbital decay", "orbit geometry"],
                        "excludes": ["binary reduction unless explicit"],
                        "related_topics": [],
                        "rules": [
                            {
                                "rule_id": "exp_orbit_decay",
                                "title": "Orbital decay energy accounting",
                                "trigger": "satellite orbital decay with drag",
                                "check_logic": "track orbital energy loss consistently",
                                "scope": "domain",
                                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["G", "M", "r"]},
                                "support": {"count": 3, "sample_ids": ["1"]},
                                "match_features": {
                                    "trigger_keywords": ["orbital", "decay"],
                                    "object_keywords": ["energy", "loss"],
                                    "scene_trigger_terms": ["satellite orbital decay"],
                                    "formula_trigger_terms": ["energy loss"],
                                    "required_symbols": ["G", "M", "r"],
                                    "weak_symbol_terms": [],
                                    "primitive": "none",
                                },
                            }
                        ],
                        "knowledge_reference": {"rule_ids": ["k1"], "keywords": ["orbit", "gravity"]},
                        "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                        "retrieval_hints": {
                            "scene_keywords": ["satellite orbital decay", "atmospheric drag"],
                            "topic_keywords": ["orbit", "gravity", "drag"],
                            "required_symbols": ["G", "M", "r"],
                        },
                        "scenario_clusters": [
                            {
                                "cluster_id": "orbital_decay_and_orbit_accounting",
                                "name": "Orbital Decay and Orbit Accounting",
                                "description": "Satellite orbit change and orbital-energy bookkeeping.",
                                "includes": ["orbital decay", "energy loss"],
                                "excludes": ["binary reduction"],
                                "entry_cues": ["satellite orbital decay", "atmospheric drag"],
                                "related_clusters": [],
                                "rule_groups": [
                                    {
                                        "group_id": "orbit_decay_core_checks",
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
                "name": "Modern Physics",
                "topics": [
                    {
                        "name": "Special Relativity (Time Dilation, Length Contraction)",
                        "description": "Relativistic observation and frame effects.",
                        "includes": ["pinhole camera observation"],
                        "excludes": ["pure classical imaging"],
                        "related_topics": [],
                        "rules": [
                            {
                                "rule_id": "exp_pinhole",
                                "title": "Pinhole simultaneity",
                                "trigger": "pinhole camera sees moving rod",
                                "check_logic": "treat exposure as simultaneous in observer frame",
                                "scope": "domain",
                                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["L", "v"]},
                                "support": {"count": 2, "sample_ids": ["2"]},
                                "match_features": {
                                    "trigger_keywords": ["pinhole", "camera"],
                                    "object_keywords": ["moving", "rod"],
                                    "scene_trigger_terms": ["pinhole camera", "moving rod"],
                                    "formula_trigger_terms": ["observer frame"],
                                    "required_symbols": ["L", "v"],
                                    "weak_symbol_terms": [],
                                    "primitive": "none",
                                },
                            }
                        ],
                        "knowledge_reference": {"rule_ids": ["r1"], "keywords": ["relativity", "length contraction"]},
                        "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                        "retrieval_hints": {
                            "scene_keywords": ["pinhole camera", "moving rod"],
                            "topic_keywords": ["relativity", "length contraction", "observer frame"],
                            "required_symbols": ["L", "v"],
                        },
                        "scenario_clusters": [
                            {
                                "cluster_id": "observation_and_projection",
                                "name": "Observation and Projection Geometry",
                                "description": "Observation tasks where camera timing and projection are central.",
                                "includes": ["pinhole camera", "observed length"],
                                "excludes": ["frequency-only problems"],
                                "entry_cues": ["pinhole camera", "moving rod"],
                                "related_clusters": [],
                                "rule_groups": [
                                    {
                                        "group_id": "projection_checks",
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
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "name": "Electromagnetism",
                    "topics": [
                        {
                            "name": "DC Circuits and Kirchhoff's Laws",
                            "description": "Static circuit solving with loop and node constraints.",
                            "includes": ["Kirchhoff equations"],
                            "excludes": ["reactive dynamics"],
                            "related_topics": [],
                            "rules": [
                                {
                                    "rule_id": f"exp_dc_{index}",
                                    "title": f"DC rule {index}",
                                    "trigger": "Kirchhoff branch current",
                                    "check_logic": f"Loop and node consistency {index}",
                                    "scope": "domain",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["I", "V"]},
                                    "support": {"count": 1, "sample_ids": [str(index)]},
                                    "match_features": {
                                        "trigger_keywords": ["Kirchhoff", "branch"],
                                        "object_keywords": ["loop", "node"],
                                        "scene_trigger_terms": ["DC circuit"],
                                        "formula_trigger_terms": ["loop equation"],
                                        "required_symbols": ["I", "V"],
                                        "weak_symbol_terms": [],
                                        "primitive": "none",
                                    },
                                }
                                for index in range(3)
                            ],
                            "knowledge_reference": {"rule_ids": ["dc1"], "keywords": ["Kirchhoff", "circuit"]},
                            "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                            "retrieval_hints": {
                                "scene_keywords": ["DC circuit"],
                                "topic_keywords": ["Kirchhoff", "loop", "node"],
                                "required_symbols": ["I", "V"],
                            },
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


if __name__ == "__main__":
    unittest.main()
