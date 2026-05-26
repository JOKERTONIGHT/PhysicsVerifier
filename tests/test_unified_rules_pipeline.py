from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from scripts.unified_rules_pipeline import (
    build_blueprint_validation_command,
    build_cluster_proposal_command,
    build_rebuild_catalog_command,
    build_rule_embedding_cluster_command,
    build_server_command,
    dataset_paths,
    run_analyze_embedding_clusters,
    run_build_blueprints,
    run_prepare_cluster,
)

TMP_ROOT = Path("results/test_tmp")
TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"physicsverifier_pipeline_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class UnifiedRulesPipelineTests(unittest.TestCase):
    def test_dataset_paths_group_outputs_by_dataset_directory(self) -> None:
        paths = dataset_paths("3000")

        self.assertEqual(paths["result_dir"], Path("results/unified_rules_3000"))
        self.assertEqual(paths["semantic"], Path("results/unified_rules_3000/semantic_experience.json"))
        self.assertEqual(paths["distilled"], Path("results/unified_rules_3000/semantic_experience_distilled.json"))
        self.assertEqual(paths["distilled_for_cluster"], Path("results/unified_rules_3000/semantic_experience_distilled_for_cluster.json"))
        self.assertEqual(paths["rule_embedding_input"], Path("results/unified_rules_3000/rule_embedding_input.json"))
        self.assertEqual(paths["rule_embedding_clusters"], Path("results/unified_rules_3000/rule_embedding_clusters.json"))
        self.assertEqual(paths["rule_embedding_cluster_report"], Path("results/unified_rules_3000/rule_embedding_cluster_report.json"))
        self.assertEqual(paths["cluster_proposals"], Path("results/unified_rules_3000/cluster_proposals.json"))
        self.assertEqual(paths["precluster_report"], Path("results/unified_rules_3000/precluster_report.json"))
        self.assertEqual(paths["catalog"], Path("catalogs/rules_unified_3000.json"))

    def test_server_command_uses_canonical_paths(self) -> None:
        command = build_server_command(dataset="3000", model="gemini-3-flash-preview-thinking")

        self.assertIn("scripts/run_semantic_experience.py", command)
        self.assertIn("results/unified_rules_3000/semantic_experience.json", command)
        self.assertIn("results/unified_rules_3000/semantic_experience_distilled.json", command)
        self.assertIn("--model gemini-3-flash-preview-thinking", command)

    def test_rule_embedding_cluster_command_uses_canonical_paths(self) -> None:
        command = build_rule_embedding_cluster_command(dataset="3000", embedding_model="text-embedding-3-large")

        self.assertIn("scripts/run_rule_embedding_clustering.py", command)
        self.assertIn("results/unified_rules_3000/rule_embedding_input.json", command)
        self.assertIn("results/unified_rules_3000/rule_embedding_clusters.json", command)
        self.assertIn("--embedding-model text-embedding-3-large", command)

    def test_cluster_proposal_command_uses_embedding_clusters_not_full_topic_rules(self) -> None:
        command = build_cluster_proposal_command(dataset="3000", model="gemini-3-flash-preview-thinking")

        self.assertIn("scripts/generate_cluster_proposals.py", command)
        self.assertIn("--embedding-clusters results/unified_rules_3000/rule_embedding_clusters.json", command)
        self.assertIn("--rule-input results/unified_rules_3000/rule_embedding_input.json", command)
        self.assertNotIn("--distilled-experience", command)

    def test_validation_and_rebuild_commands_use_generated_blueprints(self) -> None:
        validation = build_blueprint_validation_command(dataset="3000")
        rebuild = build_rebuild_catalog_command(dataset="3000")

        self.assertIn("catalogs/scenario_cluster_blueprints_generated_3000.json", validation)
        self.assertIn("--mode subset", validation)
        self.assertIn("scripts/build_unified_catalog.py", rebuild)
        self.assertIn("results/unified_rules_3000/semantic_experience_distilled_for_cluster.json", rebuild)
        self.assertIn("catalogs/scenario_cluster_blueprints_generated_3000.json", rebuild)

    def test_build_blueprints_subcommand_uses_canonical_proposal_output(self) -> None:
        root = _case_dir()
        paths = dataset_paths("mini", root=root)
        paths["result_dir"].mkdir(parents=True, exist_ok=True)
        paths["generated_blueprints"].parent.mkdir(parents=True, exist_ok=True)
        paths["cluster_proposals"].write_text(
            json.dumps(
                {
                    "proposals": [
                        {
                            "topic_key": "mechanics::kinematics",
                            "clusters": [
                                {
                                    "cluster_id": "timing_checks",
                                    "name": "Timing Checks",
                                    "summary": "Timing checks",
                                    "candidate_rule_ids": ["r1"],
                                }
                            ],
                            "residual_rule_ids": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        report = run_build_blueprints(dataset="mini", root=root)

        self.assertEqual(report["topic_count"], 1)
        self.assertTrue(paths["generated_blueprints"].exists())

    def test_analyze_embedding_subcommand_uses_canonical_outputs(self) -> None:
        root = _case_dir()
        paths = dataset_paths("mini", root=root)
        paths["result_dir"].mkdir(parents=True, exist_ok=True)
        paths["rule_embedding_clusters"].write_text(
            json.dumps(
                {
                    "topics": [
                        {
                            "topic_key": "mechanics::kinematics",
                            "rule_count": 4,
                            "clusters": [{"cluster_id": "c1", "rule_ids": ["r1", "r2"]}],
                            "residual_rule_ids": ["r3", "r4"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        report = run_analyze_embedding_clusters(dataset="mini", root=root)

        self.assertEqual(report["total_rule_count"], 4)
        self.assertTrue(paths["rule_embedding_cluster_report"].exists())

    def test_prepare_cluster_subcommand_builds_canonical_outputs(self) -> None:
        root = _case_dir()
        paths = dataset_paths("mini", root=root)
        knowledge_path = root / "knowledge.json"
        tagged_path = root / "tagged.json"
        baseline_path = root / "baseline.json"
        paths["result_dir"].mkdir(parents=True, exist_ok=True)
        paths["catalog"].parent.mkdir(parents=True, exist_ok=True)

        knowledge_path.write_text(
            json.dumps(
                {
                    "domains": [
                        {
                            "name": "Mechanics",
                            "topics": [
                                {
                                    "name": "Fluid Dynamics (Bernoulli's Equation, Continuity)",
                                    "rules": [],
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        tagged_path.write_text("[]", encoding="utf-8")
        baseline_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "catalog_type": "unified_rules_v2",
                        "schema_profile": "semantic_navigation_tree_minimal",
                        "total_executable_rules": 0,
                        "topics_with_rules": 0,
                        "total_scenario_clusters": 0,
                    },
                    "domains": [
                        {
                            "name": "Mechanics",
                            "topics": [
                                {
                                    "name": "Fluid Dynamics (Bernoulli's Equation, Continuity)",
                                    "rules": [],
                                    "scenario_clusters": [],
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths["distilled"].write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "rule_id": "r1",
                            "domain": "Mechanics",
                            "topic": "Mechanics / Fluid Dynamics (Bernoulli's Equation, Continuity)",
                            "title": "连续性方程截面积校验",
                            "trigger": "管道流体截面积变化",
                            "check_logic": "检查 A v 是否守恒。",
                            "error_type": "calculation",
                            "symbolic_hint": {
                                "primitive": "formula_pattern",
                                "canonical": "A1 v1 = A2 v2",
                                "required_symbols": ["A", "v"],
                            },
                            "auxiliary": {
                                "node_summary": "流体连续性检查",
                                "scene_cues": ["pipe"],
                                "boundary_cues": ["incompressible"],
                                "explore_cues": ["viscosity"],
                                "evidence_sample_ids": ["s1"],
                            },
                            "count": 1,
                            "sample_ids": ["s1"],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        report = run_prepare_cluster(
            dataset="mini",
            root=root,
            knowledge_path=knowledge_path,
            tagged_path=tagged_path,
            baseline_catalog_path=baseline_path,
            scenario_cluster_blueprints_paths=[],
        )

        self.assertEqual(report["catalog"]["total_executable_rules"], 1)
        self.assertTrue(paths["distilled_for_cluster"].exists())
        self.assertTrue(paths["rule_embedding_input"].exists())
        self.assertTrue(paths["catalog"].exists())
        self.assertTrue(paths["precluster_report"].exists())


if __name__ == "__main__":
    unittest.main()
