from __future__ import annotations

import unittest

from scripts.refine_cluster_blueprints import build_generated_blueprints_from_refined_proposals
from scripts.validate_cluster_blueprints import validate_blueprints_against_catalog


class RefineClusterBlueprintsTests(unittest.TestCase):
    def test_build_generated_blueprints_converts_candidate_rule_ids_to_rule_groups(self) -> None:
        proposals = {
            "proposals": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "topic_key": "mechanics::kinematics",
                    "clusters": [
                        {
                            "cluster_id": "timing_checks",
                            "name": "Timing Checks",
                            "summary": "Timing relations",
                            "description": "Timing relation checks.",
                            "includes": ["time relation"],
                            "excludes": [],
                            "entry_cues": ["time"],
                            "related_clusters": [],
                            "candidate_rule_ids": ["exp_k1", "exp_k2"],
                        }
                    ],
                    "residual_rule_ids": ["exp_k3"],
                }
            ]
        }
        blueprints = build_generated_blueprints_from_refined_proposals(proposals)
        self.assertEqual(
            blueprints["mechanics::kinematics"][0]["rule_groups"][0]["rule_ids"],
            ["exp_k1", "exp_k2"],
        )
        self.assertEqual(blueprints["mechanics::kinematics"][1]["cluster_id"], "residual_rules_01")
        self.assertEqual(blueprints["mechanics::kinematics"][1]["rule_groups"][0]["rule_ids"], ["exp_k3"])

    def test_build_generated_blueprints_splits_residual_rules_into_non_general_clusters(self) -> None:
        proposals = {
            "proposals": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "topic_key": "mechanics::kinematics",
                    "clusters": [],
                    "residual_rule_ids": [f"exp_k{i}" for i in range(1, 15)],
                }
            ]
        }

        blueprints = build_generated_blueprints_from_refined_proposals(proposals)
        residual_clusters = blueprints["mechanics::kinematics"]

        self.assertEqual([item["cluster_id"] for item in residual_clusters], ["residual_rules_01", "residual_rules_02"])
        self.assertEqual(len(residual_clusters[0]["rule_groups"][0]["rule_ids"]), 12)
        self.assertEqual(len(residual_clusters[1]["rule_groups"][0]["rule_ids"]), 2)

    def test_build_generated_blueprints_maps_navigation_cues_to_builder_compat_fields(self) -> None:
        proposals = {
            "proposals": [
                {
                    "domain": "Mechanics",
                    "topic": "Fluid Dynamics",
                    "topic_key": "mechanics::fluid dynamics",
                    "clusters": [
                        {
                            "cluster_id": "bernoulli_path",
                            "name": "Bernoulli Path",
                            "summary": "Bernoulli energy path checks",
                            "description": "Pressure-speed-height reasoning.",
                            "scene_cues": ["pipe narrowing", "height change"],
                            "boundary_cues": ["viscous loss dominates"],
                            "explore_cues": ["continuity coupling"],
                            "candidate_rule_ids": ["exp_f1"],
                        }
                    ],
                    "residual_rule_ids": [],
                }
            ]
        }

        blueprints = build_generated_blueprints_from_refined_proposals(proposals)
        cluster = blueprints["mechanics::fluid dynamics"][0]

        self.assertEqual(cluster["entry_cues"], ["pipe narrowing", "height change"])
        self.assertEqual(cluster["excludes"], ["viscous loss dominates"])
        self.assertIn("continuity coupling", cluster["description"])

    def test_validate_generated_blueprints_reports_missing_topic_coverage(self) -> None:
        catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Kinematics", "rules": [{"rule_id": "exp_k1"}]},
                        {"name": "Dynamics", "rules": [{"rule_id": "exp_d1"}]},
                    ],
                }
            ]
        }
        blueprints = {"mechanics::kinematics": []}
        report = validate_blueprints_against_catalog(catalog, blueprints, mode="full")
        self.assertIn("mechanics::dynamics", report["missing_topics"])

    def test_validate_generated_blueprints_subset_mode_ignores_missing_topics(self) -> None:
        catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Kinematics", "rules": [{"rule_id": "exp_k1"}]},
                        {"name": "Dynamics", "rules": [{"rule_id": "exp_d1"}]},
                    ],
                }
            ]
        }
        blueprints = {
            "mechanics::kinematics": [
                {
                    "cluster_id": "c1",
                    "name": "Cluster 1",
                    "description": "",
                    "includes": [],
                    "excludes": [],
                    "entry_cues": [],
                    "related_clusters": [],
                    "rule_groups": [
                        {
                            "group_id": "g1",
                            "name": "Group 1",
                            "summary": "",
                            "activation_condition": "",
                            "rule_ids": ["exp_k1"],
                        }
                    ],
                }
            ]
        }
        report = validate_blueprints_against_catalog(catalog, blueprints, mode="subset")
        self.assertTrue(report["valid"])
        self.assertEqual(report["mode"], "subset")
        self.assertEqual(report["validated_topic_count"], 1)
        self.assertEqual(report["missing_topics"], [])

    def test_validate_generated_blueprints_reports_duplicate_rule_assignments(self) -> None:
        catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [{"rule_id": "exp_k1"}, {"rule_id": "exp_k2"}],
                        }
                    ],
                }
            ]
        }
        blueprints = {
            "mechanics::kinematics": [
                {
                    "cluster_id": "c1",
                    "name": "Cluster 1",
                    "description": "",
                    "includes": [],
                    "excludes": [],
                    "entry_cues": [],
                    "related_clusters": [],
                    "rule_groups": [
                        {
                            "group_id": "g1",
                            "name": "Group 1",
                            "summary": "",
                            "activation_condition": "",
                            "rule_ids": ["exp_k1", "exp_k2"],
                        }
                    ],
                },
                {
                    "cluster_id": "c2",
                    "name": "Cluster 2",
                    "description": "",
                    "includes": [],
                    "excludes": [],
                    "entry_cues": [],
                    "related_clusters": [],
                    "rule_groups": [
                        {
                            "group_id": "g2",
                            "name": "Group 2",
                            "summary": "",
                            "activation_condition": "",
                            "rule_ids": ["exp_k2"],
                        }
                    ],
                },
            ]
        }
        report = validate_blueprints_against_catalog(catalog, blueprints)
        self.assertEqual(report["topics_with_duplicate_rule_assignments"], ["mechanics::kinematics"])


if __name__ == "__main__":
    unittest.main()
