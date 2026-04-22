from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from core.top_down_verifier import TopDownVerifier


REPO_ROOT = Path(__file__).resolve().parents[1]


class UnifiedTopDownIntegrationTests(unittest.TestCase):
    def test_runtime_verifier_uses_unified_v2_retrieval(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "description": "Basic motion relations.",
                            "includes": ["velocity", "acceleration"],
                            "excludes": [],
                            "related_topics": [],
                            "rules": [
                                {
                                    "rule_id": "exp_k_0",
                                    "title": "Kinematics rule 0",
                                    "trigger": "velocity acceleration",
                                    "check_logic": "displacement velocity time",
                                    "error_type": "logic",
                                    "scope": "domain",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["v", "t"]},
                                    "support": {"count": 1, "sample_ids": ["0"]},
                                    "match_features": {
                                        "trigger_keywords": ["velocity", "acceleration"],
                                        "object_keywords": ["displacement", "time"],
                                        "scene_trigger_terms": ["straight-line motion"],
                                        "formula_trigger_terms": ["displacement"],
                                        "required_symbols": ["v", "t"],
                                        "weak_symbol_terms": [],
                                        "primitive": "none",
                                    },
                                }
                            ],
                            "knowledge_reference": {"rule_ids": ["k1"], "keywords": ["velocity", "displacement"]},
                            "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                            "retrieval_hints": {
                                "scene_keywords": ["velocity"],
                                "topic_keywords": ["velocity", "displacement"],
                                "required_symbols": ["v", "t"],
                            },
                            "scenario_clusters": [],
                        }
                    ],
                }
            ],
        }
        sample = {
            "id": "sample_runtime_1",
            "question": "A particle moves with velocity and acceleration. Find displacement as a function of time.",
            "prediction": "Use velocity and acceleration over time.",
            "answer": "",
        }

        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        test_dir = results_dir / "_unified_v2_runtime_test"
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            catalog_path = test_dir / "rules_unified.json"
            catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")

            verifier = TopDownVerifier(
                llm_model=None,
                unified_rules_path=str(catalog_path),
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_agentic_postcheck=False,
            )
            result = verifier.verify(sample)

            self.assertTrue(verifier._unified_mode)
            self.assertTrue(verifier._unified_v2_mode)
            self.assertEqual(result["selection_strategy"], "semantic_unavailable")
            self.assertEqual(result["semantic_selection_error"], "Semantic matcher is not available.")
            self.assertEqual(result["verifier"], "top_down_rule_based")
            self.assertEqual(result["retrieved_topics"], [])
            self.assertEqual(result["retrieved_clusters"], [])
            self.assertEqual(result["retrieved_rules"], [])
            self.assertEqual(len(verifier.rule_verifier.rules_to_check), 0)
            self.assertIsNone(result["topic"])
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_top_down_verifier_prefers_semantic_tree_path_when_available(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
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
                                    "error_type": "logic",
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
                }
            ],
        }
        sample = {
            "id": "sample_semantic_runtime_1",
            "question": "A pinhole camera observes a rod moving with velocity v.",
            "prediction": "Treat the exposure as simultaneous in the observer frame.",
            "answer": "",
        }

        class _FakeSemanticMatcher:
            available = True

            def select_tree_semantically(self, sample_payload, catalog_payload):
                topic_obj = catalog_payload["domains"][0]["topics"][0]
                rule_obj = topic_obj["rules"][0]
                return {
                    "selected_domains": ["Modern Physics"],
                    "selected_topics": [
                        {
                            "domain": "Modern Physics",
                            "topic": "Special Relativity (Time Dilation, Length Contraction)",
                            "score": 0.9,
                            "reason": "directly about relativistic observation",
                            "topic_obj": topic_obj,
                        }
                    ],
                    "selected_clusters": [
                        {
                            "domain": "Modern Physics",
                            "topic": "Special Relativity (Time Dilation, Length Contraction)",
                            "cluster_id": "observation_and_projection",
                            "cluster": "Observation and Projection Geometry",
                            "score": 0.88,
                            "reason": "camera observation cluster",
                            "cluster_obj": topic_obj["scenario_clusters"][0],
                            "topic_obj": topic_obj,
                            "rule_groups": topic_obj["scenario_clusters"][0]["rule_groups"],
                            "rule_ids": ["exp_pinhole"],
                            "topic_rules": {"exp_pinhole": rule_obj},
                        }
                    ],
                    "selected_rules": [
                        {
                            "domain": "Modern Physics",
                            "topic": "Special Relativity (Time Dilation, Length Contraction)",
                            "cluster_id": "observation_and_projection",
                            "cluster": "Observation and Projection Geometry",
                            "score": 0.93,
                            "reason": "matches pinhole observation",
                            "rule_obj": rule_obj,
                        }
                    ],
                }

        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        test_dir = results_dir / "_unified_v2_semantic_runtime_test"
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            catalog_path = test_dir / "rules_unified.json"
            catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")

            verifier = TopDownVerifier(
                llm_model="fake-model",
                unified_rules_path=str(catalog_path),
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_agentic_postcheck=False,
            )
            verifier.semantic_matcher = _FakeSemanticMatcher()
            verifier.rule_verifier.analyze = lambda _: {"diagnostics": []}
            result = verifier.verify(sample)

            self.assertEqual(result["selection_strategy"], "semantic_tree_selection")
            self.assertEqual(result["semantic_selection_error"], "")
            self.assertEqual(result["topic"], "Special Relativity (Time Dilation, Length Contraction)")
            self.assertEqual(len(result["retrieved_topics"]), 1)
            self.assertEqual(len(result["retrieved_clusters"]), 1)
            self.assertEqual(len(result["retrieved_rules"]), 1)
            self.assertEqual(result["retrieved_clusters"][0]["cluster_id"], "observation_and_projection")
            self.assertEqual(result["retrieved_rules"][0]["rule_id"], "exp_pinhole")
            self.assertEqual(result["verifier"], "unified_v2_semantic_rule_based")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_top_down_verifier_marks_semantic_path_even_when_no_rules_selected(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "name": "Electromagnetism",
                    "topics": [
                        {
                            "name": "Current, Resistance, and Ohm's Law",
                            "description": "Ohmic transport and resistive dissipation.",
                            "includes": ["wire resistance", "current"],
                            "excludes": [],
                            "related_topics": [],
                            "rules": [],
                            "knowledge_reference": {"rule_ids": ["c1"], "keywords": ["resistance", "current"]},
                            "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                            "retrieval_hints": {
                                "scene_keywords": ["wire resistance"],
                                "topic_keywords": ["current", "resistance"],
                                "required_symbols": ["I", "R"],
                            },
                            "scenario_clusters": [],
                        }
                    ],
                }
            ],
        }
        sample = {
            "id": "sample_semantic_runtime_2",
            "question": "Find the resistive loss in a transmission wire.",
            "prediction": "Use current and resistance.",
            "answer": "",
        }

        class _FakeSemanticMatcher:
            available = True

            def select_tree_semantically(self, sample_payload, catalog_payload):
                topic_obj = catalog_payload["domains"][0]["topics"][0]
                return {
                    "selected_domains": ["Electromagnetism"],
                    "selected_topics": [
                        {
                            "domain": "Electromagnetism",
                            "topic": "Current, Resistance, and Ohm's Law",
                            "score": 0.95,
                            "reason": "resistive-loss problem",
                            "topic_obj": topic_obj,
                        }
                    ],
                    "selected_clusters": [],
                    "selected_rules": [],
                }

        results_dir = REPO_ROOT / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        test_dir = results_dir / "_unified_v2_semantic_no_rules_test"
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            catalog_path = test_dir / "rules_unified.json"
            catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")

            verifier = TopDownVerifier(
                llm_model="fake-model",
                unified_rules_path=str(catalog_path),
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_agentic_postcheck=False,
            )
            verifier.semantic_matcher = _FakeSemanticMatcher()
            result = verifier.verify(sample)

            self.assertEqual(result["selection_strategy"], "semantic_tree_selection")
            self.assertEqual(result["semantic_selection_error"], "")
            self.assertEqual(result["verifier"], "unified_v2_semantic_rule_based")
            self.assertEqual(result["retrieved_topics"][0]["topic"], "Current, Resistance, and Ohm's Law")
            self.assertEqual(result["retrieved_rules"], [])
            self.assertEqual(result["diagnostics"], [])
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
