from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from scripts.unified_rules_pipeline import (
    build_blueprint_validation_command,
    build_cluster_proposal_command,
    build_rebuild_catalog_command,
    build_runtime_eval_command,
    build_rule_embedding_cluster_command,
    build_server_command,
    dataset_paths,
    quality_report_exit_code,
    run_analyze_embedding_clusters,
    run_build_blueprints,
    run_prepare_cluster,
    run_quality_report,
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
        self.assertEqual(paths["quality_report"], Path("results/unified_rules_3000/rules_unified_quality_report.json"))
        self.assertEqual(paths["runtime_eval"], Path("results/unified_rules_3000/top_down_runtime_eval.json"))
        self.assertEqual(paths["cluster_proposals"], Path("results/unified_rules_3000/cluster_proposals.json"))
        self.assertEqual(paths["generated_blueprints"], Path("results/unified_rules_3000/cluster_blueprints_generated.json"))
        self.assertEqual(paths["precluster_report"], Path("results/unified_rules_3000/precluster_report.json"))
        self.assertEqual(paths["catalog"], Path("catalogs/rules_unified_3000.json"))

    def test_server_command_uses_canonical_paths(self) -> None:
        command = build_server_command(dataset="3000", model="gemini-3-flash-preview-thinking")

        self.assertIn("scripts/run_semantic_experience.py", command)
        self.assertIn("results/unified_rules_3000/semantic_experience.json", command)
        self.assertIn("results/unified_rules_3000/semantic_experience_distilled.json", command)
        self.assertIn("--model gemini-3-flash-preview-thinking", command)

    def test_rule_embedding_cluster_command_uses_canonical_paths(self) -> None:
        command = build_rule_embedding_cluster_command(
            dataset="3000",
            embedding_model="text-embedding-3-large",
            similarity_threshold=0.74,
        )

        self.assertIn("scripts/run_rule_embedding_clustering.py", command)
        self.assertIn("results/unified_rules_3000/rule_embedding_input.json", command)
        self.assertIn("results/unified_rules_3000/rule_embedding_clusters.json", command)
        self.assertIn("--embedding-model text-embedding-3-large", command)
        self.assertIn("--similarity-threshold 0.74", command)

    def test_cluster_proposal_command_uses_embedding_clusters_not_full_topic_rules(self) -> None:
        command = build_cluster_proposal_command(dataset="3000", model="gemini-3-flash-preview-thinking")

        self.assertIn("scripts/generate_cluster_proposals.py", command)
        self.assertIn("--embedding-clusters results/unified_rules_3000/rule_embedding_clusters.json", command)
        self.assertIn("--rule-input results/unified_rules_3000/rule_embedding_input.json", command)
        self.assertIn("--resume", command)
        self.assertIn("--continue-on-error", command)
        self.assertNotIn("--distilled-experience", command)

    def test_validation_and_rebuild_commands_use_generated_blueprints(self) -> None:
        validation = build_blueprint_validation_command(dataset="3000")
        rebuild = build_rebuild_catalog_command(dataset="3000")

        self.assertIn("results/unified_rules_3000/cluster_blueprints_generated.json", validation)
        self.assertIn("--mode subset", validation)
        self.assertIn("scripts/build_unified_catalog.py", rebuild)
        self.assertIn("results/unified_rules_3000/semantic_experience_distilled_for_cluster.json", rebuild)
        self.assertIn("results/unified_rules_3000/cluster_blueprints_generated.json", rebuild)

    def test_runtime_eval_command_uses_canonical_catalog_and_output(self) -> None:
        command = build_runtime_eval_command(
            dataset="3000",
            samples="data/evaluation_sample_debug_30.json",
            limit=0,
            sample_ids="170364,157816",
            output="results/unified_rules_3000/top_down_runtime_eval_targeted.json",
        )

        self.assertIn("scripts/evaluate_top_down_runtime.py", command)
        self.assertIn("--samples data/evaluation_sample_debug_30.json", command)
        self.assertIn("--catalog catalogs/rules_unified_3000.json", command)
        self.assertIn("--output results/unified_rules_3000/top_down_runtime_eval_targeted.json", command)
        self.assertIn("--limit 0", command)
        self.assertIn("--sample-ids 170364,157816", command)

    def test_runtime_eval_command_defaults_to_no_limit_for_targeted_samples(self) -> None:
        command = build_runtime_eval_command(
            dataset="3000",
            samples="data/evaluation_sample_debug_30.json",
            sample_ids="170364,157816,142965,147128",
            output="results/unified_rules_3000/top_down_runtime_eval_targeted.json",
        )

        self.assertIn("--output results/unified_rules_3000/top_down_runtime_eval_targeted.json", command)
        self.assertIn("--limit 0", command)
        self.assertIn("--sample-ids 170364,157816,142965,147128", command)

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

    def test_quality_report_subcommand_uses_canonical_outputs(self) -> None:
        root = _case_dir()
        paths = dataset_paths("mini", root=root)
        paths["result_dir"].mkdir(parents=True, exist_ok=True)
        paths["catalog"].parent.mkdir(parents=True, exist_ok=True)
        paths["catalog"].write_text(
            json.dumps(
                {
                    "metadata": {
                        "catalog_type": "unified_rules_v2",
                        "schema_profile": "semantic_navigation_tree_minimal",
                    },
                    "domains": [
                        {
                            "id": "mechanics",
                            "name": "Mechanics",
                            "summary": "Mechanics domain.",
                            "topics": [],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        paths["cluster_proposals"].write_text(
            json.dumps({"proposals": [], "failures": []}),
            encoding="utf-8",
        )
        paths["runtime_eval"].write_text(
            json.dumps(
                {
                    "summary": {"sample_count": 1, "semantic_error_count": 0, "rule_selection_rate": 1.0, "average_selected_rules": 1.0},
                    "rows": [
                        {
                            "sample_id": "s1",
                            "selection_strategy": "semantic_tree_selection",
                            "semantic_selection_error": "",
                            "topic_count": 1,
                            "cluster_count": 1,
                            "rule_count": 1,
                            "diagnostic_count": 1,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        report = run_quality_report(dataset="mini", root=root)

        self.assertEqual(report["schema"]["schema_profile"], "semantic_navigation_tree_minimal")
        self.assertTrue(report["runtime_eval"]["available"])
        self.assertFalse(report["runtime_eval"]["stale"])
        self.assertTrue(paths["quality_report"].exists())

    def test_quality_report_can_use_targeted_runtime_eval_override(self) -> None:
        root = _case_dir()
        paths = dataset_paths("mini", root=root)
        paths["result_dir"].mkdir(parents=True, exist_ok=True)
        paths["catalog"].parent.mkdir(parents=True, exist_ok=True)
        paths["catalog"].write_text(
            json.dumps(
                {
                    "metadata": {
                        "catalog_type": "unified_rules_v2",
                        "schema_profile": "semantic_navigation_tree_minimal",
                    },
                    "domains": [],
                }
            ),
            encoding="utf-8",
        )
        paths["cluster_proposals"].write_text(
            json.dumps({"proposals": [], "failures": []}),
            encoding="utf-8",
        )
        targeted_runtime = paths["result_dir"] / "top_down_runtime_eval_targeted.json"
        targeted_output = paths["result_dir"] / "rules_unified_quality_report_targeted.json"
        targeted_runtime.write_text(
            json.dumps(
                {
                    "summary": {"sample_count": 1, "semantic_error_count": 0, "rule_selection_rate": 1.0, "average_selected_rules": 5.0},
                    "rows": [{"sample_id": "s1", "topic_count": 1, "cluster_count": 1, "rule_count": 5, "semantic_selection_error": ""}],
                }
            ),
            encoding="utf-8",
        )

        report = run_quality_report(
            dataset="mini",
            root=root,
            runtime_eval_path=targeted_runtime,
            output_path=targeted_output,
        )

        self.assertTrue(report["overall"]["readiness_gates"]["runtime_rule_cap_respected"])
        self.assertEqual(report["runtime_eval"]["sample_count"], 1)
        self.assertTrue(targeted_output.exists())

    def test_quality_report_fail_on_blocking_gates_returns_nonzero_exit_code(self) -> None:
        report = {
            "overall": {
                "blocking_gate_count": 1,
                "readiness_gates": {"runtime_rule_cap_respected": False},
            }
        }

        self.assertEqual(quality_report_exit_code(report, fail_on_blocking_gates=True), 1)
        self.assertEqual(quality_report_exit_code(report, fail_on_blocking_gates=False), 0)

    def test_quality_report_fail_on_blocking_gates_returns_zero_when_gates_pass(self) -> None:
        report = {
            "overall": {
                "blocking_gate_count": 0,
                "readiness_gates": {"runtime_rule_cap_respected": True},
            }
        }

        self.assertEqual(quality_report_exit_code(report, fail_on_blocking_gates=True), 0)

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
