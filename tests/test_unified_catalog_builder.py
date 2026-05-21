from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from scripts.build_unified_catalog import (
    build_unified_catalog,
    build_unified_catalog_from_data,
    merge_scenario_cluster_blueprints,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SAMPLE_PATH = REPO_ROOT / "data" / "evaluation_sample_300.json"


class UnifiedRulesV2RepositoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = build_unified_catalog(
            knowledge_path=REPO_ROOT / "catalogs" / "rules_catalog_top_down.json",
            distilled_path=REPO_ROOT / "catalogs" / "semantic_experience_distilled_300.json",
            tagged_path=REPO_ROOT / "catalogs" / "rules_300_tagged.json",
        )
        cls.eval_sample_index = {
            str(item.get("id")): item
            for item in json.loads(EVAL_SAMPLE_PATH.read_text(encoding="utf-8"))
            if isinstance(item, dict)
        }

    def test_repository_build_preserves_all_knowledge_topics(self) -> None:
        meta = self.catalog["metadata"]
        self.assertEqual(meta["version"], "2.0")
        self.assertEqual(meta["catalog_type"], "unified_rules_v2")
        self.assertEqual(meta["schema_profile"], "semantic_navigation_tree_minimal")
        self.assertEqual(meta["total_domains"], 6)
        self.assertEqual(meta["total_topics"], 123)
        self.assertEqual(meta["topics_with_rules"], 70)
        self.assertEqual(meta["total_executable_rules"], 514)
        self.assertEqual(meta["knowledge_rule_references"], 1082)

    def test_catalog_stores_only_navigation_tree_fields(self) -> None:
        removed_topic_fields = {
            "knowledge_reference",
            "tagged_reference",
            "retrieval_hints",
            "description",
            "includes",
            "excludes",
            "related_topics",
        }
        removed_rule_fields = {
            "scope",
            "support",
            "match_features",
            "retrieval_flags",
            "applicability",
            "negative_cues",
        }
        removed_cluster_fields = {
            "cluster_id",
            "description",
            "includes",
            "excludes",
            "entry_cues",
            "related_clusters",
        }

        for domain in self.catalog["domains"]:
            self.assertEqual(set(domain.keys()), {"id", "name", "summary", "topics"})
            for topic in domain["topics"]:
                self.assertFalse(removed_topic_fields.intersection(topic.keys()))
                self.assertTrue({"id", "name", "summary", "scenario_clusters", "rules"}.issubset(topic.keys()))
                for cluster in topic["scenario_clusters"]:
                    self.assertFalse(removed_cluster_fields.intersection(cluster.keys()))
                    self.assertTrue({"id", "name", "summary", "rule_ids"}.issubset(cluster.keys()))
                for rule in topic["rules"]:
                    self.assertFalse(removed_rule_fields.intersection(rule.keys()))

    def test_executable_rules_are_distilled_navigation_leaves(self) -> None:
        required_rule_keys = {
            "rule_id",
            "title",
            "summary",
            "trigger",
            "check_logic",
            "error_type",
            "symbolic_hint",
        }
        total_rules = 0
        for domain in self.catalog["domains"]:
            for topic in domain["topics"]:
                self.assertIn("scenario_clusters", topic)
                self.assertIn("summary", topic)
                self.assertNotIn("includes", topic)
                self.assertNotIn("excludes", topic)
                for rule in topic["rules"]:
                    total_rules += 1
                    self.assertTrue(required_rule_keys.issubset(rule.keys()))
                    self.assertNotIn("source", rule)
                    self.assertNotIn("source_file", rule)
                    self.assertTrue(str(rule["rule_id"]).startswith("exp_"))
        self.assertEqual(total_rules, self.catalog["metadata"]["total_executable_rules"])

    def test_focus_topics_receive_semantic_scenario_clusters(self) -> None:
        topics = {
            (domain["name"], topic["name"]): topic
            for domain in self.catalog["domains"]
            for topic in domain["topics"]
        }

        gravitation = topics[("Mechanics", "Gravitation and Kepler's Laws")]
        gravitation_cluster_ids = {cluster["id"] for cluster in gravitation["scenario_clusters"]}
        self.assertIn("orbital_decay_and_orbit_accounting", gravitation_cluster_ids)

        relativity = topics[("Modern Physics", "Special Relativity (Time Dilation, Length Contraction)")]
        relativity_cluster_ids = {cluster["id"] for cluster in relativity["scenario_clusters"]}
        self.assertIn("observation_and_projection", relativity_cluster_ids)

        induction = topics[("Electromagnetism", "Electromagnetic Induction and Faraday's Law")]
        induction_cluster_ids = {cluster["id"] for cluster in induction["scenario_clusters"]}
        self.assertIn("motional_emf_and_rotation", induction_cluster_ids)

        snell = topics[("Optics", "Snell's Law and Critical Angle")]
        snell_cluster_ids = {cluster["id"] for cluster in snell["scenario_clusters"]}
        self.assertIn("refractive_gradient_and_mirage", snell_cluster_ids)

        kinematics = topics[("Mechanics", "Kinematics in 1D/2D/3D")]
        kinematics_cluster_ids = {cluster["id"] for cluster in kinematics["scenario_clusters"]}
        self.assertIn("timing_and_displacement_relations", kinematics_cluster_ids)

        first_law = topics[("Thermodynamics & Statistical Physics", "First Law of Thermodynamics")]
        first_law_cluster_ids = {cluster["id"] for cluster in first_law["scenario_clusters"]}
        self.assertIn("energy_bookkeeping_and_sign_conventions", first_law_cluster_ids)

        rotational = topics[("Mechanics", "Rotational Kinematics and Dynamics")]
        rotational_cluster_ids = {cluster["id"] for cluster in rotational["scenario_clusters"]}
        self.assertIn("torque_balance_and_angular_acceleration", rotational_cluster_ids)

        biot_savart = topics[("Electromagnetism", "Biot-Savart Law and Ampere's Law")]
        biot_cluster_ids = {cluster["id"] for cluster in biot_savart["scenario_clusters"]}
        self.assertIn("symmetry_loop_and_field_direction", biot_cluster_ids)

class UnifiedRulesV2UnitTests(unittest.TestCase):
    def test_multiple_blueprint_sources_merge_by_topic_key(self) -> None:
        merged = merge_scenario_cluster_blueprints(
            {
                "mechanics::kinematics": [
                    {
                        "cluster_id": "seed_cluster",
                        "name": "Seed Cluster",
                        "description": "",
                        "includes": [],
                        "excludes": [],
                        "entry_cues": [],
                        "related_clusters": [],
                        "rule_groups": [],
                    }
                ]
            },
            {
                "mechanics::kinematics": [
                    {
                        "cluster_id": "generated_cluster",
                        "name": "Generated Cluster",
                        "description": "",
                        "includes": [],
                        "excludes": [],
                        "entry_cues": [],
                        "related_clusters": [],
                        "rule_groups": [],
                    }
                ]
            },
        )
        self.assertEqual(
            [cluster["cluster_id"] for cluster in merged["mechanics::kinematics"]],
            ["seed_cluster", "generated_cluster"],
        )

    def test_generic_math_rule_is_reclassified_for_retrieval(self) -> None:
        knowledge = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [{"name": "Projectile Motion", "rules": []}],
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

        catalog = build_unified_catalog_from_data(knowledge, distilled, [])
        rule = catalog["domains"][0]["topics"][0]["rules"][0]
        self.assertEqual(rule["summary"], "三角恒等式代换一致性: 方程中同时出现 sinθ, cosθ 且含有根式")
        self.assertNotIn("negative_cues", rule)
        self.assertNotIn("applicability", rule)
        self.assertNotIn("match_features", rule)
        self.assertNotIn("retrieval_flags", rule)

    def test_unmatched_distilled_topic_raises(self) -> None:
        knowledge = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [{"id": "kin_001", "title": "Kinematics rule", "description": "", "check_logic": ""}],
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

        with self.assertRaises(ValueError):
            build_unified_catalog_from_data(knowledge, distilled, [])

    def test_build_unified_catalog_accepts_multiple_blueprint_paths(self) -> None:
        knowledge = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [{"id": "kin_001", "title": "Kinematics rule", "description": "", "check_logic": ""}],
                        }
                    ],
                }
            ]
        }
        distilled = {
            "rules": [
                {
                    "rule_id": "exp_k1",
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": "Timing relation",
                    "trigger": "timing relation",
                    "check_logic": "check timing relation",
                    "error_type": "logic",
                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
                    "count": 1,
                    "sample_ids": ["1"],
                },
                {
                    "rule_id": "exp_k2",
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "title": "Projection relation",
                    "trigger": "projection relation",
                    "check_logic": "check projection relation",
                    "error_type": "logic",
                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
                    "count": 1,
                    "sample_ids": ["2"],
                }
            ]
        }
        tagged = []
        seed_blueprints = {
            "mechanics::kinematics": [
                {
                    "cluster_id": "seed_cluster",
                    "name": "Seed Cluster",
                    "description": "Seed cluster.",
                    "includes": [],
                    "excludes": [],
                    "entry_cues": [],
                    "related_clusters": [],
                    "rule_groups": [
                        {
                            "group_id": "seed_cluster_rules",
                            "name": "Seed Cluster Rules",
                            "summary": "Seed rules",
                            "activation_condition": "Use for seed coverage.",
                            "match_any": ["timing"],
                        }
                    ],
                }
            ]
        }
        generated_blueprints = {
            "mechanics::kinematics": [
                {
                    "cluster_id": "generated_cluster",
                    "name": "Generated Cluster",
                    "description": "Generated cluster.",
                    "includes": [],
                    "excludes": [],
                    "entry_cues": [],
                    "related_clusters": [],
                    "rule_groups": [
                        {
                            "group_id": "generated_cluster_rules",
                            "name": "Generated Cluster Rules",
                            "summary": "Generated rules",
                            "activation_condition": "Use for generated coverage.",
                            "rule_ids": ["exp_k2"],
                        }
                    ],
                }
            ]
        }

        temp_root = REPO_ROOT / "results" / "_builder_multi_blueprint_test"
        if temp_root.exists():
            shutil.rmtree(temp_root, ignore_errors=True)
        temp_root.mkdir(parents=True, exist_ok=True)
        try:
            knowledge_path = temp_root / "knowledge.json"
            distilled_path = temp_root / "distilled.json"
            tagged_path = temp_root / "tagged.json"
            seed_path = temp_root / "seed.json"
            generated_path = temp_root / "generated.json"
            knowledge_path.write_text(json.dumps(knowledge, ensure_ascii=False), encoding="utf-8")
            distilled_path.write_text(json.dumps(distilled, ensure_ascii=False), encoding="utf-8")
            tagged_path.write_text(json.dumps(tagged, ensure_ascii=False), encoding="utf-8")
            seed_path.write_text(json.dumps(seed_blueprints, ensure_ascii=False), encoding="utf-8")
            generated_path.write_text(json.dumps(generated_blueprints, ensure_ascii=False), encoding="utf-8")

            catalog = build_unified_catalog(
                knowledge_path=knowledge_path,
                distilled_path=distilled_path,
                tagged_path=tagged_path,
                scenario_cluster_blueprints_paths=[seed_path, generated_path],
            )
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        topic = catalog["domains"][0]["topics"][0]
        self.assertEqual(
            [cluster["id"] for cluster in topic["scenario_clusters"]],
            ["seed_cluster", "generated_cluster"],
        )


if __name__ == "__main__":
    unittest.main()
