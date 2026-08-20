from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from core.physics_rule_verifier import PhysicsRuleVerifier
from scripts.analyze_rule_matching import analyze_matching
from rule_framework.builder import (
    CLUSTER_BUCKET_THRESHOLD,
    CLUSTER_TOPIC_THRESHOLD,
    build_unified_catalog,
    build_unified_catalog_from_data,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _semantic_runtime_catalog() -> dict:
    rule = {
        "rule_id": "exp_pinhole",
        "title": "Pinhole simultaneity",
        "trigger": "pinhole camera sees moving rod",
        "check_logic": "treat exposure as simultaneous in observer frame",
        "error_type": "logic",
        "scope": "domain",
        "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["L", "v"]},
        "match_features": {
            "trigger_keywords": ["pinhole", "camera"],
            "object_keywords": ["moving", "rod"],
            "required_symbols": ["L", "v"],
            "primitive": "none",
        },
    }
    cluster = {
        "id": "observation_and_projection",
        "name": "Observation and Projection Geometry",
        "summary": "Camera timing and projection in relativistic observation.",
        "rule_groups": [],
        "rule_ids": ["exp_pinhole"],
    }
    return {
        "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
        "domains": [
            {
                "name": "Modern Physics",
                "topics": [
                    {
                        "name": "Special Relativity",
                        "summary": "Relativistic observation and frame effects.",
                        "rules": [rule],
                        "scenario_clusters": [cluster],
                    }
                ],
            }
        ],
    }


class _FakeSemanticMatcher:
    available = True

    def __init__(self, rule_score=0.93):
        self.rule_score = float(rule_score)

    def select_tree_semantically(self, sample_payload, catalog_payload):
        topic = catalog_payload["domains"][0]["topics"][0]
        cluster = topic["scenario_clusters"][0]
        rule = topic["rules"][0]
        return {
            "input_policy": "background_navigation_prediction_rule_only",
            "domain_judgments": [
                {"domain": "Modern Physics", "score": 0.96, "reason": "relativistic observation"}
            ],
            "selected_domains": ["Modern Physics"],
            "selected_topics": [
                {
                    "domain": "Modern Physics",
                    "topic": "Special Relativity",
                    "score": 0.94,
                    "reason": "moving-rod observation",
                    "topic_obj": topic,
                }
            ],
            "selected_clusters": [
                {
                    "domain": "Modern Physics",
                    "topic": "Special Relativity",
                    "cluster_id": "observation_and_projection",
                    "cluster": "Observation and Projection Geometry",
                    "score": 0.92,
                    "reason": "camera observation cluster",
                    "cluster_obj": cluster,
                }
            ],
            "selected_rules": [
                {
                    "domain": "Modern Physics",
                    "topic": "Special Relativity",
                    "cluster_id": "observation_and_projection",
                    "cluster": "Observation and Projection Geometry",
                    "score": self.rule_score,
                    "reason": "directly applicable pinhole rule",
                    "rule_obj": rule,
                }
            ],
        }


class _UnavailableSemanticMatcher:
    available = False

    def select_tree_semantically(self, sample_payload, catalog_payload):
        raise AssertionError("unavailable matcher must not be called")


class _EmptyRuleSemanticMatcher(_FakeSemanticMatcher):
    def select_tree_semantically(self, sample_payload, catalog_payload):
        result = super().select_tree_semantically(sample_payload, catalog_payload)
        result["selected_rules"] = []
        return result


class _RecordingExperienceCodeEngine:
    available = True

    def __init__(self, topic_rule_ids):
        self.topic_rule_ids = list(topic_rule_ids)
        self.list_topic_calls = []
        self.run_calls = []

    def list_topic_rule_ids(self, domain, topic):
        self.list_topic_calls.append((domain, topic))
        return list(self.topic_rule_ids)

    def run_rule(self, rule_id, sample):
        self.run_calls.append(rule_id)
        return {
            "result": "fail",
            "message": f"generated check failed for {rule_id}",
            "evidence": "exposure as simultaneous",
        }


class UnifiedRulesV2RepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = build_unified_catalog(
            knowledge_path=REPO_ROOT / "catalogs" / "rules_catalog_top_down.json",
            distilled_path=REPO_ROOT / "catalogs" / "semantic_experience_distilled_300.json",
            tagged_path=REPO_ROOT / "catalogs" / "rules_300_tagged.json",
        )

    def test_repository_build_preserves_all_knowledge_topics(self) -> None:
        meta = self.catalog["metadata"]
        self.assertEqual(meta["version"], "2.0")
        self.assertEqual(meta["catalog_type"], "unified_rules_v2")
        self.assertEqual(meta["total_domains"], 6)
        self.assertEqual(meta["total_topics"], 123)
        self.assertEqual(meta["topics_with_rules"], 70)
        self.assertEqual(meta["total_executable_rules"], 514)
        self.assertEqual(meta["knowledge_rule_references"], 1082)

    def test_executable_rules_are_distilled_only(self) -> None:
        required_rule_keys = {
            "rule_id",
            "title",
            "trigger",
            "check_logic",
            "error_type",
            "scope",
            "symbolic_hint",
            "support",
            "match_features",
            "retrieval_text",
        }
        total_rules = 0
        for domain in self.catalog["domains"]:
            for topic in domain["topics"]:
                self.assertIn("knowledge_reference", topic)
                self.assertIn("tagged_reference", topic)
                self.assertIn("retrieval_hints", topic)
                self.assertIn("clusters", topic)
                for rule in topic["rules"]:
                    total_rules += 1
                    self.assertTrue(required_rule_keys.issubset(rule.keys()))
                    self.assertNotIn("source", rule)
                    self.assertNotIn("source_file", rule)
                    self.assertTrue(str(rule["rule_id"]).startswith("exp_"))
                    self.assertIn("count", rule["support"])
                    self.assertIn("sample_ids", rule["support"])
                    self.assertIn(rule["scope"], {"domain", "meta"})
                    self.assertEqual(
                        set(rule["match_features"].keys()),
                        {
                            "trigger_keywords",
                            "object_keywords",
                            "required_symbols",
                            "primitive",
                            "match_text_normalized",
                            "negative_keywords",
                        },
                    )
        self.assertEqual(total_rules, 514)

    def test_clusters_follow_threshold_policy(self) -> None:
        for domain in self.catalog["domains"]:
            for topic in domain["topics"]:
                rules = topic["rules"]
                clusters = topic["clusters"]
                rule_index = {rule["rule_id"]: rule for rule in rules}

                if len(rules) < CLUSTER_TOPIC_THRESHOLD:
                    self.assertEqual(clusters, [])
                    continue

                for cluster in clusters:
                    self.assertGreaterEqual(len(cluster["rule_ids"]), CLUSTER_BUCKET_THRESHOLD)
                    self.assertTrue(cluster["keywords"])
                    for rule_id in cluster["rule_ids"]:
                        self.assertIn(rule_id, rule_index)
                        self.assertEqual(rule_index[rule_id]["error_type"], cluster["error_type"])


class UnifiedRulesV2UnitTests(unittest.TestCase):
    def _run_semantic_symbolic_case(self, *, catalog, matcher, engine, suffix):
        sample = {
            "id": f"sample_{suffix}",
            "question": "A pinhole camera observes a rod moving with velocity v.",
            "prediction": "Treat the exposure as simultaneous in the observer frame.",
            "answer": "",
        }
        test_dir = REPO_ROOT / "results" / f"_unified_v2_{suffix}_test"
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            catalog_path = test_dir / "rules_unified.json"
            catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
            verifier = PhysicsRuleVerifier(
                llm_model=None,
                unified_rules_path=str(catalog_path),
                semantic_matcher=matcher,
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_symbolic_check=True,
            )
            verifier.experience_code_engine = engine
            verifier.semantic_checker.analyze = lambda _: {"diagnostics": []}
            return verifier.verify(sample)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_manual_override_relabels_generic_trig_rule(self) -> None:
        knowledge = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Projectile Motion",
                            "rules": [],
                        }
                    ],
                }
            ]
        }
        distilled = {
            "rules": [
                {
                    "rule_id": "exp_8f7c1ad1fb477295",
                    "domain": "Mechanics",
                    "topic": "Projectile Motion",
                    "title": "三角恒等式代换一致性",
                    "trigger": "方程中同时出现 sinθ, cosθ 且含有根式",
                    "check_logic": "统一化为 tanθ 的表达形式",
                    "error_type": "calculation",
                    "symbolic_hint": {"primitive": "formula_pattern", "canonical": "", "required_symbols": ["θ"]},
                    "count": 1,
                    "sample_ids": ["8332"],
                }
            ]
        }
        tagged = []

        catalog = build_unified_catalog_from_data(knowledge, distilled, tagged)
        rule = catalog["domains"][0]["topics"][0]["rules"][0]
        self.assertEqual(rule["scope"], "meta")
        self.assertEqual(rule["manual_override_reason"], "generic_trig_substitution_rule")
        self.assertNotIn("sin", rule["match_features"]["trigger_keywords"])
        self.assertNotIn("cos", rule["match_features"]["trigger_keywords"])

    def test_rule_scene_phrases_improve_topic_anchors(self) -> None:
        knowledge = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Projectile Motion", "rules": []},
                        {"name": "Gravitation and Kepler's Laws", "rules": []},
                    ],
                }
            ]
        }
        distilled = {
            "rules": [
                {
                    "rule_id": "exp_proj",
                    "domain": "Mechanics",
                    "topic": "Projectile Motion",
                    "title": "Elevated launch tangent substitution",
                    "trigger": "tan theta root equation in projectile launch",
                    "check_logic": "check tangent substitution",
                    "error_type": "calculation",
                    "symbolic_hint": {"primitive": "formula_pattern", "canonical": "", "required_symbols": ["θ"]},
                    "count": 1,
                    "sample_ids": ["1"],
                },
                {
                    "rule_id": "exp_orbit",
                    "domain": "Mechanics",
                    "topic": "Gravitation and Kepler's Laws",
                    "title": "Orbital Decay energy accounting",
                    "trigger": "satellite orbital decay with atmospheric drag",
                    "check_logic": "check total mechanical energy change",
                    "error_type": "logic",
                    "symbolic_hint": {"primitive": "equation_equivalence", "canonical": "", "required_symbols": ["G", "M", "r"]},
                    "count": 1,
                    "sample_ids": ["2"],
                },
            ]
        }
        tagged = []

        catalog = build_unified_catalog_from_data(knowledge, distilled, tagged)
        topics = {topic["name"]: topic for domain in catalog["domains"] for topic in domain["topics"]}
        grav_scene = topics["Gravitation and Kepler's Laws"]["retrieval_hints"]["scene_keywords"]
        self.assertIn("Orbital Decay energy accounting", grav_scene)
        self.assertIn("satellite orbital decay", " ".join(grav_scene).lower())

        sample = [
            {
                "id": "sample_orbit",
                "question": "A satellite undergoes orbital decay because of atmospheric drag as its orbit radius decreases.",
                "prediction": "Track the total mechanical energy loss during orbital decay.",
                "answer": "",
            }
        ]
        analysis = analyze_matching(catalog, sample, top_topics=2, top_rules=6, annotation_limit=1)
        self.assertEqual(analysis["per_sample"][0]["retrieved_topics"][0]["topic"], "Gravitation and Kepler's Laws")

    def test_manual_topic_anchor_override_boosts_special_relativity(self) -> None:
        knowledge = {
            "domains": [
                {
                    "name": "Modern Physics",
                    "topics": [
                        {"name": "Special Relativity (Time Dilation, Length Contraction)", "rules": []},
                    ],
                }
            ]
        }
        distilled = {
            "rules": [
                {
                    "rule_id": "exp_relativity_pin",
                    "domain": "Modern Physics",
                    "topic": "Special Relativity (Time Dilation, Length Contraction)",
                    "title": "针孔快门同时测量判定",
                    "trigger": "出现 opening the pinhole for a very short time 或 pinhole camera 观察高速杆",
                    "check_logic": "视为观测者系同时测量，直接适用长度收缩",
                    "error_type": "logic",
                    "symbolic_hint": {"primitive": "equation_equivalence", "canonical": "", "required_symbols": ["L", "v"]},
                    "count": 1,
                    "sample_ids": ["1"],
                }
            ]
        }
        catalog = build_unified_catalog_from_data(knowledge, distilled, [])
        topic = catalog["domains"][0]["topics"][0]
        self.assertIn("pinhole camera", topic["retrieval_hints"]["scene_keywords"])
        self.assertIn("rod", topic["retrieval_hints"]["topic_keywords"])

        sample = [
            {
                "id": "sample_relativity",
                "question": "A pinhole camera observes a rod in motion and asks for the apparent observed length.",
                "prediction": "Treat it as simultaneous measurement in the observer frame.",
                "answer": "",
            }
        ]
        analysis = analyze_matching(catalog, sample, top_topics=1, top_rules=6, annotation_limit=1)
        self.assertEqual(
            analysis["per_sample"][0]["retrieved_topics"][0]["topic"],
            "Special Relativity (Time Dilation, Length Contraction)",
        )

    def test_induction_anchor_override_beats_generic_resistance_topic(self) -> None:
        knowledge = {
            "domains": [
                {
                    "name": "Electromagnetism",
                    "topics": [
                        {"name": "Current, Resistance, and Ohm's Law", "rules": []},
                        {"name": "Electromagnetic Induction and Faraday's Law", "rules": []},
                    ],
                }
            ]
        }
        distilled = {
            "rules": [
                {
                    "rule_id": "exp_resistor",
                    "domain": "Electromagnetism",
                    "topic": "Current, Resistance, and Ohm's Law",
                    "title": "Resistive wire loss",
                    "trigger": "voltage current resistor wire loss",
                    "check_logic": "apply Ohm law consistently",
                    "error_type": "logic",
                    "symbolic_hint": {"primitive": "equation_equivalence", "canonical": "", "required_symbols": ["R", "I", "V"]},
                    "count": 1,
                    "sample_ids": ["1"],
                },
                {
                    "rule_id": "exp_induction",
                    "domain": "Electromagnetism",
                    "topic": "Electromagnetic Induction and Faraday's Law",
                    "title": "Eddy current magnetic braking",
                    "trigger": "rotating disc in magnetic field with foucault current",
                    "check_logic": "use induced emf from flux change",
                    "error_type": "logic",
                    "symbolic_hint": {"primitive": "equation_equivalence", "canonical": "", "required_symbols": ["B", "R"]},
                    "count": 1,
                    "sample_ids": ["2"],
                },
            ]
        }
        catalog = build_unified_catalog_from_data(knowledge, distilled, [])
        topics = {topic["name"]: topic for domain in catalog["domains"] for topic in domain["topics"]}
        self.assertIn("eddy current", topics["Electromagnetic Induction and Faraday's Law"]["retrieval_hints"]["scene_keywords"])
        self.assertNotIn("Resistance", topics["Current, Resistance, and Ohm's Law"]["retrieval_hints"]["scene_keywords"])

        sample = [
            {
                "id": "sample_eddy",
                "question": "Foucault currents produce magnetic braking on a rotating metal disc in a magnetic field.",
                "prediction": "Relate the induced emf to the changing flux and eddy current dissipation.",
                "answer": "",
            }
        ]
        analysis = analyze_matching(catalog, sample, top_topics=2, top_rules=6, annotation_limit=1)
        self.assertEqual(
            analysis["per_sample"][0]["retrieved_topics"][0]["topic"],
            "Electromagnetic Induction and Faraday's Law",
        )

    def test_unmatched_distilled_topic_collects_orphans(self) -> None:
        knowledge = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [
                                {"id": "kin_001", "title": "Kinematics rule", "description": "", "check_logic": ""}
                            ],
                        }
                    ],
                }
            ]
        }
        distilled = {
            "rules": [
                {
                    "rule_id": "exp_bad",
                    "domain": "Mechanics",
                    "topic": "Mechanics / Dynamics",
                    "title": "Bad topic",
                    "trigger": "something",
                    "check_logic": "something else",
                    "error_type": "logic",
                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
                    "count": 1,
                    "sample_ids": ["1"],
                }
            ]
        }
        tagged = []

        catalog = build_unified_catalog_from_data(knowledge, distilled, tagged)
        meta = catalog["metadata"]
        self.assertGreaterEqual(int(meta.get("distilled_orphan_count") or 0), 1)
        orphans = meta.get("distilled_orphan_rules") or []
        self.assertTrue(any(str(o.get("rule_id")) == "exp_bad" for o in orphans))

    def test_offline_matching_respects_top_k_and_top_n(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [
                                {
                                    "rule_id": f"exp_k_{idx}",
                                    "title": f"Kinematics rule {idx}",
                                    "trigger": "velocity acceleration",
                                    "check_logic": "displacement velocity time",
                                    "error_type": "logic",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["v", "t"]},
                                    "support": {"count": idx + 1, "sample_ids": [str(idx)]},
                                    "match_features": {
                                        "trigger_keywords": ["velocity", "acceleration"],
                                        "object_keywords": ["displacement", "time"],
                                        "required_symbols": ["v", "t"],
                                        "primitive": "none",
                                    },
                                }
                                for idx in range(4)
                            ],
                            "knowledge_reference": {"rule_ids": ["k1"], "keywords": ["velocity", "displacement"]},
                            "tagged_reference": {"source_ids": ["t1"], "titles": ["速度判断"], "aliases": ["速度判断"], "keywords": ["速度"]},
                            "retrieval_hints": {
                                "scene_keywords": ["velocity"],
                                "topic_keywords": ["velocity", "displacement"],
                                "required_symbols": ["v", "t"],
                            },
                            "clusters": [],
                        },
                        {
                            "name": "Dynamics",
                            "rules": [
                                {
                                    "rule_id": f"exp_d_{idx}",
                                    "title": f"Dynamics rule {idx}",
                                    "trigger": "force mass",
                                    "check_logic": "acceleration Newton",
                                    "error_type": "logic",
                                    "scope": "domain",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["F", "a"]},
                                    "support": {"count": idx + 2, "sample_ids": [str(idx)]},
                                    "match_features": {
                                        "trigger_keywords": ["force", "mass"],
                                        "object_keywords": ["acceleration", "Newton"],
                                        "required_symbols": ["F", "a"],
                                        "primitive": "none",
                                    },
                                }
                                for idx in range(4)
                            ],
                            "knowledge_reference": {"rule_ids": ["d1"], "keywords": ["force", "mass"]},
                            "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                            "retrieval_hints": {
                                "scene_keywords": ["force"],
                                "topic_keywords": ["force", "mass"],
                                "required_symbols": ["F", "a"],
                            },
                            "clusters": [],
                        },
                        {
                            "name": "Oscillations",
                            "rules": [],
                            "knowledge_reference": {"rule_ids": ["o1"], "keywords": ["oscillation"]},
                            "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                            "retrieval_hints": {"scene_keywords": ["oscillation"], "topic_keywords": ["oscillation"], "required_symbols": ["T"]},
                            "clusters": [],
                        },
                        {
                            "name": "Gravitation",
                            "rules": [],
                            "knowledge_reference": {"rule_ids": ["g1"], "keywords": ["gravity"]},
                            "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                            "retrieval_hints": {"scene_keywords": ["gravity"], "topic_keywords": ["gravity"], "required_symbols": ["G"]},
                            "clusters": [],
                        },
                    ],
                }
            ],
        }
        samples = [
            {
                "id": "sample_1",
                "question": "A particle moves with velocity and acceleration. Find displacement as a function of time.",
                "prediction": "Use velocity and acceleration over time.",
                "answer": "",
            }
        ]

        analysis = analyze_matching(catalog, samples, top_topics=3, top_rules=6, annotation_limit=1)
        sample_result = analysis["per_sample"][0]
        self.assertLessEqual(len(sample_result["retrieved_topics"]), 3)
        self.assertLessEqual(len(sample_result["retrieved_rules"]), 6)
        self.assertEqual(sample_result["candidate_rule_count"], len(sample_result["retrieved_rules"]))
        self.assertTrue(sample_result["retrieved_topics"])
        self.assertEqual(sample_result["retrieved_topics"][0]["topic"], "Kinematics")
        self.assertIn("rule_topk_saturation_ratio", analysis["summary"])
        self.assertIn("average_positive_rule_trace_count", analysis["summary"])
        self.assertIn("strong_top1_cross_topic_ratio", analysis["summary"])
        self.assertIn("meta_rule_count", analysis["per_sample"][0])

    def test_runtime_verifier_uses_unified_v2_retrieval(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [
                                {
                                    "rule_id": f"exp_k_{idx}",
                                    "title": f"Kinematics rule {idx}",
                                    "trigger": "velocity acceleration",
                                    "check_logic": "displacement velocity time",
                                    "error_type": "logic",
                                    "scope": "domain",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["v", "t"]},
                                    "support": {"count": idx + 1, "sample_ids": [str(idx)]},
                                    "match_features": {
                                        "trigger_keywords": ["velocity", "acceleration"],
                                        "object_keywords": ["displacement", "time"],
                                        "required_symbols": ["v", "t"],
                                        "primitive": "none",
                                    },
                                }
                                for idx in range(4)
                            ],
                            "knowledge_reference": {"rule_ids": ["k1"], "keywords": ["velocity", "displacement"]},
                            "tagged_reference": {"source_ids": ["t1"], "titles": ["速度判断"], "aliases": ["速度判断"], "keywords": ["速度"]},
                            "retrieval_hints": {
                                "scene_keywords": ["velocity"],
                                "topic_keywords": ["velocity", "displacement"],
                                "required_symbols": ["v", "t"],
                            },
                            "clusters": [],
                        },
                        {
                            "name": "Dynamics",
                            "rules": [
                                {
                                    "rule_id": f"exp_d_{idx}",
                                    "title": f"Dynamics rule {idx}",
                                    "trigger": "force mass",
                                    "check_logic": "acceleration Newton",
                                    "error_type": "logic",
                                    "scope": "domain",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["F", "a"]},
                                    "support": {"count": idx + 2, "sample_ids": [str(idx)]},
                                    "match_features": {
                                        "trigger_keywords": ["force", "mass"],
                                        "object_keywords": ["acceleration", "Newton"],
                                        "required_symbols": ["F", "a"],
                                        "primitive": "none",
                                    },
                                }
                                for idx in range(4)
                            ],
                            "knowledge_reference": {"rule_ids": ["d1"], "keywords": ["force", "mass"]},
                            "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                            "retrieval_hints": {
                                "scene_keywords": ["force"],
                                "topic_keywords": ["force", "mass"],
                                "required_symbols": ["F", "a"],
                            },
                            "clusters": [],
                        },
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

            verifier = PhysicsRuleVerifier(
                llm_model=None,
                unified_rules_path=str(catalog_path),
                unified_retrieval_mode="lexical",
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_agentic_postcheck=False,
            )
            result = verifier.verify(sample)

            self.assertTrue(verifier._unified_mode)
            self.assertTrue(verifier._unified_v2_mode)
            self.assertEqual(result["verifier"], "unified_v2_rule_based")
            self.assertLessEqual(len(result["retrieved_topics"]), 3)
            self.assertLessEqual(len(result["retrieved_rules"]), 6)
            self.assertEqual(len(verifier.semantic_checker.rules_to_check), len(result["retrieved_rules"]))
            self.assertTrue(result["retrieved_topics"])
            self.assertTrue(result["retrieved_rules"])
            self.assertEqual(result["topic"], "Kinematics")
            self.assertEqual(result["retrieval_score_kind"], "lexical")
            self.assertEqual(result["retrieved_rules"][0]["score_kind"], "lexical")
            self.assertEqual(result["retrieved_rules"][0]["scope"], "domain")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_runtime_verifier_uses_injected_semantic_tree_by_default(self) -> None:
        catalog = _semantic_runtime_catalog()
        sample = {
            "id": "sample_semantic_runtime",
            "question": "A pinhole camera observes a rod moving with velocity v.",
            "prediction": "Treat the exposure as simultaneous in the observer frame.",
            "answer": "",
        }
        test_dir = REPO_ROOT / "results" / "_unified_v2_semantic_runtime_test"
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            catalog_path = test_dir / "rules_unified.json"
            catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
            verifier = PhysicsRuleVerifier(
                llm_model=None,
                unified_rules_path=str(catalog_path),
                semantic_matcher=_FakeSemanticMatcher(),
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_symbolic_check=False,
            )
            verifier.semantic_checker.analyze = lambda _: {"diagnostics": []}

            def unexpected_lexical(*args, **kwargs):
                raise AssertionError("semantic default must not call lexical retrieval")

            verifier._retrieve_unified_v2_topics = unexpected_lexical
            verifier._retrieve_unified_v2_rules = unexpected_lexical
            result = verifier.verify(sample)

            self.assertEqual(verifier.unified_retrieval_mode, "semantic")
            self.assertEqual(result["selection_strategy"], "semantic_tree_selection")
            self.assertEqual(result["semantic_selection_error"], "")
            self.assertEqual(result["verifier"], "unified_v2_semantic_rule_based")
            self.assertEqual(result["topic"], "Special Relativity")
            self.assertEqual(result["retrieved_clusters"][0]["cluster_id"], "observation_and_projection")
            self.assertEqual(result["retrieved_rules"][0]["rule_id"], "exp_pinhole")
            self.assertEqual(result["retrieval_score_kind"], "semantic_0_1")
            self.assertEqual(result["retrieved_rules"][0]["score_kind"], "semantic_0_1")
            self.assertEqual(result["retrieved_rules"][0]["publish_gate"]["score_kind"], "semantic_0_1")
            self.assertEqual(verifier.semantic_checker.rules_to_check, ["exp_pinhole"])
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_public_semantic_retrieval_trace_does_not_call_checker_or_symbolic(self) -> None:
        catalog = _semantic_runtime_catalog()
        sample = {
            "id": "sample_semantic_retrieval_only",
            "question": "A pinhole camera observes a rod moving with velocity v.",
            "prediction": "Treat the exposure as simultaneous in the observer frame.",
            "answer": "",
        }
        test_dir = REPO_ROOT / "results" / "_unified_v2_semantic_retrieval_only_test"
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            catalog_path = test_dir / "rules_unified.json"
            catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
            verifier = PhysicsRuleVerifier(
                llm_model=None,
                unified_rules_path=str(catalog_path),
                semantic_matcher=_FakeSemanticMatcher(),
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_symbolic_check=True,
            )

            def unexpected_checker(_):
                raise AssertionError("retrieval-only must not call Semantic Checker")

            class UnexpectedSymbolicEngine:
                available = True

                def run_rule(self, *args, **kwargs):
                    raise AssertionError("retrieval-only must not run generated symbolic checks")

                def list_topic_rule_ids(self, *args, **kwargs):
                    raise AssertionError("retrieval-only must not enumerate symbolic checks")

            verifier.semantic_checker.analyze = unexpected_checker
            verifier.experience_code_engine = UnexpectedSymbolicEngine()
            checker_rules_before = list(verifier.semantic_checker.rules_to_check)
            trace = verifier.retrieve_unified_semantic_tree(sample)

            self.assertEqual(trace["id"], sample["id"])
            self.assertEqual(trace["verifier"], "unified_v2_semantic_retrieval_only")
            self.assertEqual(trace["selection_strategy"], "semantic_tree_selection")
            self.assertEqual(trace["topic"], "Special Relativity")
            self.assertEqual(trace["retrieved_domains"][0]["domain"], "Modern Physics")
            self.assertEqual(trace["retrieved_topics"][0]["topic"], "Special Relativity")
            self.assertEqual(trace["retrieved_clusters"][0]["cluster_id"], "observation_and_projection")
            self.assertEqual(trace["retrieved_rules"][0]["rule_id"], "exp_pinhole")
            self.assertNotIn("diagnostics", trace)
            self.assertEqual(verifier.semantic_checker.rules_to_check, checker_rules_before)
            json.dumps(trace, ensure_ascii=False)

            verifier.unified_retrieval_mode = "lexical"
            with self.assertRaisesRegex(ValueError, "requires unified_retrieval_mode='semantic'"):
                verifier.retrieve_unified_semantic_tree(sample)
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_semantic_min_publish_score_keeps_trace_but_blocks_execution(self) -> None:
        catalog = _semantic_runtime_catalog()
        sample = {
            "id": "sample_semantic_publish_threshold",
            "question": "A pinhole camera observes a rod moving with velocity v.",
            "prediction": "Treat the exposure as simultaneous in the observer frame.",
            "answer": "",
        }
        engine = _RecordingExperienceCodeEngine(["exp_pinhole"])
        test_dir = REPO_ROOT / "results" / "_unified_v2_semantic_publish_threshold_test"
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            catalog_path = test_dir / "rules_unified.json"
            catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
            verifier = PhysicsRuleVerifier(
                llm_model=None,
                unified_rules_path=str(catalog_path),
                semantic_matcher=_FakeSemanticMatcher(rule_score=0.93),
                semantic_min_publish_score=0.95,
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_symbolic_check=True,
            )
            verifier.experience_code_engine = engine
            verifier.semantic_checker.rules_to_check = []

            def unexpected_checker(_):
                raise AssertionError("below-threshold semantic rule must not enter the checker")

            verifier.semantic_checker.analyze = unexpected_checker
            result = verifier.verify(sample)

            self.assertEqual(result["semantic_min_publish_score"], 0.95)
            self.assertEqual([item["rule_id"] for item in result["retrieved_rules"]], ["exp_pinhole"])
            gate = result["retrieved_rules"][0]["publish_gate"]
            self.assertFalse(gate["publishable"])
            self.assertIn("below_semantic_publish_score", gate["reasons"])
            self.assertEqual(gate["min_publish_score"], 0.95)
            self.assertEqual(verifier.semantic_checker.rules_to_check, [])
            self.assertEqual(engine.list_topic_calls, [])
            self.assertEqual(engine.run_calls, [])
            self.assertEqual(result["experience_code_post_diagnostics"], [])
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_unavailable_semantic_matcher_does_not_fall_back_to_lexical(self) -> None:
        catalog = _semantic_runtime_catalog()
        sample = {
            "id": "sample_semantic_unavailable",
            "question": "A pinhole camera observes a moving rod.",
            "prediction": "Use length contraction.",
            "answer": "",
        }
        test_dir = REPO_ROOT / "results" / "_unified_v2_semantic_unavailable_test"
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            catalog_path = test_dir / "rules_unified.json"
            catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
            verifier = PhysicsRuleVerifier(
                llm_model=None,
                unified_rules_path=str(catalog_path),
                semantic_matcher=_UnavailableSemanticMatcher(),
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_symbolic_check=False,
            )

            def unexpected_lexical(*args, **kwargs):
                raise AssertionError("unavailable semantic matcher must not trigger lexical fallback")

            verifier._retrieve_unified_v2_topics = unexpected_lexical
            verifier._retrieve_unified_v2_rules = unexpected_lexical
            result = verifier.verify(sample)

            self.assertEqual(result["selection_strategy"], "semantic_unavailable")
            self.assertEqual(result["semantic_failed_stage"], "initialization")
            self.assertIn("not available", result["semantic_selection_error"])
            self.assertEqual(result["retrieved_topics"], [])
            self.assertEqual(result["retrieved_clusters"], [])
            self.assertEqual(result["retrieved_rules"], [])
            self.assertEqual(result["verifier"], "unified_v2_semantic_rule_based")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_run_batch_fail_fast_keeps_failure_trace_and_allows_semantic_empty(self) -> None:
        catalog = _semantic_runtime_catalog()

        class ErrorMatcher:
            available = True

            def __init__(self):
                self.calls = []

            def select_tree_semantically(self, sample_payload, catalog_payload):
                self.calls.append(str(sample_payload.get("id")))
                raise RuntimeError("fake API failure")

        class EmptyMatcher:
            available = True

            def __init__(self):
                self.calls = []

            def select_tree_semantically(self, sample_payload, catalog_payload):
                self.calls.append(str(sample_payload.get("id")))
                return {
                    "domain_judgments": [],
                    "selected_domains": [],
                    "selected_topics": [],
                    "selected_clusters": [],
                    "selected_rules": [],
                }

        test_dir = REPO_ROOT / "results" / "_unified_v2_semantic_fail_fast_test"
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            catalog_path = test_dir / "rules_unified.json"
            catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")
            samples = [
                {"id": "first", "question": "q1", "prediction": "p1", "answer": ""},
                {"id": "second", "question": "q2", "prediction": "p2", "answer": ""},
            ]

            error_matcher = ErrorMatcher()
            verifier = PhysicsRuleVerifier(
                llm_model=None,
                unified_rules_path=str(catalog_path),
                semantic_matcher=error_matcher,
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_symbolic_check=False,
            )
            partial = verifier.run_batch(
                samples,
                progress_interval=0,
                fail_fast_on_semantic_error=True,
            )
            self.assertEqual(error_matcher.calls, ["first"])
            self.assertEqual(len(partial), 1)
            self.assertEqual(partial[0]["id"], "first")
            self.assertEqual(partial[0]["selection_strategy"], "semantic_error")
            self.assertEqual(partial[0]["semantic_failed_stage"], "tree")
            self.assertIn("fake API failure", partial[0]["semantic_selection_error"])

            empty_matcher = EmptyMatcher()
            verifier = PhysicsRuleVerifier(
                llm_model=None,
                unified_rules_path=str(catalog_path),
                semantic_matcher=empty_matcher,
                log_dir=str(test_dir),
                results_dir=str(test_dir),
                enable_symbolic_check=False,
            )
            completed = verifier.run_batch(
                samples,
                progress_interval=0,
                fail_fast_on_semantic_error=True,
            )
            self.assertEqual(empty_matcher.calls, ["first", "second"])
            self.assertEqual(len(completed), 2)
            self.assertEqual(
                [item["selection_strategy"] for item in completed],
                ["semantic_tree_empty", "semantic_tree_empty"],
            )
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_semantic_bottom_up_runs_only_selected_publishable_rules(self) -> None:
        catalog = _semantic_runtime_catalog()
        topic = catalog["domains"][0]["topics"][0]
        unselected_rule = dict(topic["rules"][0])
        unselected_rule.update(
            {
                "rule_id": "exp_unselected",
                "title": "Unselected neighboring rule",
            }
        )
        topic["rules"].append(unselected_rule)
        engine = _RecordingExperienceCodeEngine(["exp_pinhole", "exp_unselected"])

        result = self._run_semantic_symbolic_case(
            catalog=catalog,
            matcher=_FakeSemanticMatcher(),
            engine=engine,
            suffix="semantic_bottom_up_selected_only",
        )

        self.assertEqual(engine.list_topic_calls, [])
        self.assertEqual(engine.run_calls, ["exp_pinhole"])
        self.assertEqual(
            [item["rule_id"] for item in result["experience_code_post_diagnostics"]],
            ["exp_pinhole"],
        )
        self.assertEqual(
            [item["experience_code"]["rule_id"] for item in result["experience_post_diagnostics"]],
            ["exp_pinhole"],
        )

    def test_semantic_tree_empty_does_not_sweep_topic_experience_code(self) -> None:
        engine = _RecordingExperienceCodeEngine(["exp_pinhole"])

        result = self._run_semantic_symbolic_case(
            catalog=_semantic_runtime_catalog(),
            matcher=_EmptyRuleSemanticMatcher(),
            engine=engine,
            suffix="semantic_bottom_up_empty",
        )

        self.assertEqual(result["selection_strategy"], "semantic_tree_empty")
        self.assertEqual(engine.list_topic_calls, [])
        self.assertEqual(engine.run_calls, [])
        self.assertEqual(result["experience_code_post_diagnostics"], [])
        self.assertEqual(result["experience_post_diagnostics"], [])

    def test_semantic_bottom_up_skips_selected_rule_blocked_by_static_gate(self) -> None:
        catalog = _semantic_runtime_catalog()
        catalog["domains"][0]["topics"][0]["rules"][0]["publishable"] = False
        engine = _RecordingExperienceCodeEngine(["exp_pinhole"])

        result = self._run_semantic_symbolic_case(
            catalog=catalog,
            matcher=_FakeSemanticMatcher(),
            engine=engine,
            suffix="semantic_bottom_up_static_gate",
        )

        self.assertFalse(result["retrieved_rules"][0]["publish_gate"]["publishable"])
        self.assertEqual(engine.list_topic_calls, [])
        self.assertEqual(engine.run_calls, [])
        self.assertEqual(result["experience_code_post_diagnostics"], [])
        self.assertEqual(result["experience_post_diagnostics"], [])

    def test_meta_rules_are_deprioritized_inside_same_topic(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [
                                {
                                    "rule_id": "exp_domain",
                                    "title": "Velocity direction consistency",
                                    "trigger": "velocity displacement",
                                    "check_logic": "direction and sign should match",
                                    "error_type": "logic",
                                    "scope": "domain",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["v", "x"]},
                                    "support": {"count": 2, "sample_ids": ["1"]},
                                    "match_features": {
                                        "trigger_keywords": ["velocity", "displacement"],
                                        "object_keywords": ["direction", "sign"],
                                        "required_symbols": ["v", "x"],
                                        "primitive": "none",
                                    },
                                },
                                {
                                    "rule_id": "exp_meta",
                                    "title": "图表数据提取强制性",
                                    "trigger": "graph figure",
                                    "check_logic": "read graph before solving",
                                    "error_type": "logic",
                                    "scope": "meta",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["x"]},
                                    "support": {"count": 3, "sample_ids": ["2"]},
                                    "match_features": {
                                        "trigger_keywords": ["graph", "figure"],
                                        "object_keywords": ["read graph"],
                                        "required_symbols": ["x"],
                                        "primitive": "none",
                                    },
                                },
                            ],
                            "knowledge_reference": {"rule_ids": ["k1"], "keywords": ["velocity", "displacement"]},
                            "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                            "retrieval_hints": {
                                "scene_keywords": ["velocity"],
                                "topic_keywords": ["velocity", "displacement"],
                                "required_symbols": ["v", "x"],
                            },
                            "clusters": [],
                        }
                    ],
                }
            ],
        }
        samples = [
            {
                "id": "sample_meta_1",
                "question": "Velocity and displacement have opposite directions in the student's derivation.",
                "prediction": "The graph is not necessary here; compare sign and direction directly.",
                "answer": "",
            }
        ]

        analysis = analyze_matching(catalog, samples, top_topics=3, top_rules=6, annotation_limit=1)
        retrieved_rules = analysis["per_sample"][0]["retrieved_rules"]
        self.assertEqual(retrieved_rules[0]["rule_id"], "exp_domain")
        self.assertEqual(retrieved_rules[0]["scope"], "domain")

    def test_strong_top1_topic_blocks_foreign_generic_rule(self) -> None:
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [
                                {
                                    "rule_id": "exp_kine",
                                    "title": "Velocity direction consistency",
                                    "trigger": "velocity displacement acceleration",
                                    "check_logic": "direction and sign should match",
                                    "error_type": "logic",
                                    "scope": "domain",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["v", "x"]},
                                    "support": {"count": 4, "sample_ids": ["1"]},
                                    "match_features": {
                                        "trigger_keywords": ["velocity", "displacement", "acceleration"],
                                        "object_keywords": ["direction", "sign"],
                                        "required_symbols": ["v", "x"],
                                        "primitive": "none",
                                    },
                                }
                            ],
                            "knowledge_reference": {"rule_ids": ["k1"], "keywords": ["velocity", "displacement", "acceleration"]},
                            "tagged_reference": {"source_ids": [], "titles": ["straight-line motion"], "aliases": ["straight-line motion"], "keywords": ["velocity"]},
                            "retrieval_hints": {
                                "scene_keywords": ["straight-line motion", "velocity"],
                                "topic_keywords": ["velocity", "displacement", "acceleration"],
                                "required_symbols": ["v", "x"],
                            },
                            "clusters": [],
                        },
                        {
                            "name": "Projectile Motion",
                            "rules": [
                                {
                                    "rule_id": "exp_foreign_generic",
                                    "title": "三角恒等式代换一致性",
                                    "trigger": "sin cos tan root",
                                    "check_logic": "tan substitution",
                                    "error_type": "calculation",
                                    "scope": "meta",
                                    "manual_override_reason": "generic_trig_substitution_rule",
                                    "symbolic_hint": {"primitive": "formula_pattern", "canonical": "", "required_symbols": ["θ"]},
                                    "support": {"count": 3, "sample_ids": ["2"]},
                                    "match_features": {
                                        "trigger_keywords": ["sin", "cos", "tan", "root"],
                                        "object_keywords": ["tan substitution", "trigonometric identity"],
                                        "required_symbols": ["θ"],
                                        "primitive": "formula_pattern",
                                    },
                                }
                            ],
                            "knowledge_reference": {"rule_ids": ["p1"], "keywords": ["projectile", "launch"]},
                            "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                            "retrieval_hints": {
                                "scene_keywords": ["projectile", "launch angle"],
                                "topic_keywords": ["projectile", "parabola"],
                                "required_symbols": ["θ"],
                            },
                            "clusters": [],
                        },
                    ],
                }
            ],
        }
        samples = [
            {
                "id": "sample_strong_top1",
                "question": "A body moves in straight-line motion with velocity, displacement, and acceleration all discussed explicitly.",
                "prediction": "The student also writes sin theta, cos theta, tan theta, and a root term, but the real issue is the velocity direction sign.",
                "answer": "",
            }
        ]

        analysis = analyze_matching(catalog, samples, top_topics=2, top_rules=6, annotation_limit=1)
        sample_result = analysis["per_sample"][0]
        self.assertGreaterEqual(sample_result["topic_score_margin"], 3.0)
        self.assertFalse(sample_result["rules_outside_top1_topic"])
        self.assertEqual(sample_result["retrieved_rules"][0]["rule_id"], "exp_kine")
        self.assertEqual(analysis["summary"]["strong_top1_cross_topic_count"], 0)


if __name__ == "__main__":
    unittest.main()
