from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from core.top_down_verifier import TopDownVerifier
from scripts.analyze_unified_matching import analyze_matching
from scripts.merge_rules import (
    CLUSTER_BUCKET_THRESHOLD,
    CLUSTER_TOPIC_THRESHOLD,
    build_unified_catalog,
    build_unified_catalog_from_data,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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
                        {"trigger_keywords", "object_keywords", "required_symbols", "primitive"},
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

    def test_unmatched_distilled_topic_raises(self) -> None:
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

        with self.assertRaises(ValueError):
            build_unified_catalog_from_data(knowledge, distilled, tagged)

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
            self.assertEqual(result["verifier"], "unified_v2_rule_based")
            self.assertLessEqual(len(result["retrieved_topics"]), 3)
            self.assertLessEqual(len(result["retrieved_rules"]), 6)
            self.assertEqual(len(verifier.rule_verifier.rules_to_check), len(result["retrieved_rules"]))
            self.assertTrue(result["retrieved_topics"])
            self.assertTrue(result["retrieved_rules"])
            self.assertEqual(result["topic"], "Kinematics")
            self.assertEqual(result["retrieved_rules"][0]["scope"], "domain")
        finally:
            shutil.rmtree(test_dir, ignore_errors=True)

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
