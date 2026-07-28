from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.unified_rules_pipeline import (
    build_blueprint_validation_command,
    build_catalog_structure_validation_command,
    build_cluster_proposal_command,
    build_generalization_command,
    build_rebuild_catalog_command,
    build_rule_embedding_cluster_command,
    build_server_command,
    dataset_paths,
    quality_report_exit_code,
    run_analyze_embedding_clusters,
    run_build_blueprints,
    run_prepare_candidates,
    run_prepare_cluster,
    run_prepare_generalized,
    run_quality_report,
)

def _case_dir(test_case: unittest.TestCase) -> Path:
    directory = tempfile.TemporaryDirectory(prefix="physicsverifier_pipeline_test_")
    test_case.addCleanup(directory.cleanup)
    return Path(directory.name)


class UnifiedRulesPipelineTests(unittest.TestCase):
    def test_dataset_paths_group_outputs_by_dataset_directory(self) -> None:
        paths = dataset_paths("3000")

        self.assertEqual(paths["result_dir"], Path("results/unified_rules_3000"))
        self.assertEqual(paths["semantic"], Path("results/unified_rules_3000/semantic_experience.json"))
        self.assertEqual(paths["distilled"], Path("results/unified_rules_3000/semantic_experience_distilled.json"))
        self.assertEqual(paths["distilled_for_cluster"], Path("results/unified_rules_3000/semantic_experience_distilled_for_cluster.json"))
        self.assertEqual(paths["generalized"], Path("results/unified_rules_3000/semantic_experience_generalized.json"))
        self.assertEqual(paths["generalized_for_cluster"], Path("results/unified_rules_3000/semantic_experience_generalized_for_cluster.json"))
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

        self.assertIn("scripts/generate_experience_rules.py", command)
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
        self.assertIn("--cache results/unified_rules_3000/rule_embedding_cache.json", command)

    def test_formal_rule_embedding_command_uses_formal_rules(self) -> None:
        command = build_rule_embedding_cluster_command(dataset="3000", formal=True)

        self.assertIn("results/unified_rules_3000/formal_rule_embedding_input.json", command)
        self.assertIn("results/unified_rules_3000/formal_rule_embedding_clusters.json", command)
        self.assertIn("results/unified_rules_3000/formal_rule_embedding_cache.json", command)

    def test_generalization_command_uses_candidate_artifacts_and_safe_resume(self) -> None:
        command = build_generalization_command(dataset="3000")

        self.assertIn("semantic_experience_distilled_for_cluster.json", command)
        self.assertIn("rule_embedding_clusters.json", command)
        self.assertIn("semantic_experience_generalized.json", command)
        self.assertIn("--model gemini-3-flash-preview-nothinking", command)
        self.assertIn("--fallback-model gemini-2.5-flash-nothinking", command)
        self.assertIn("--max-clusters 0", command)
        self.assertIn("--request-timeout 120", command)
        self.assertIn("--resume", command)
        self.assertIn("--continue-on-error", command)

    def test_cluster_proposal_command_uses_embedding_clusters_not_full_topic_rules(self) -> None:
        command = build_cluster_proposal_command(dataset="3000", model="gemini-3-flash-preview-thinking")

        self.assertIn("scripts/generate_cluster_proposals.py", command)
        self.assertIn("--embedding-clusters results/unified_rules_3000/formal_rule_embedding_clusters.json", command)
        self.assertIn("--rule-input results/unified_rules_3000/formal_rule_embedding_input.json", command)
        self.assertIn("--resume", command)
        self.assertIn("--continue-on-error", command)
        self.assertNotIn("--distilled-experience", command)

    def test_validation_and_rebuild_commands_use_generated_blueprints(self) -> None:
        validation = build_blueprint_validation_command(dataset="3000")
        rebuild = build_rebuild_catalog_command(dataset="3000")

        self.assertIn("results/unified_rules_3000/cluster_blueprints_generated.json", validation)
        self.assertIn("--mode full", validation)
        self.assertIn("scripts/build_unified_catalog.py", rebuild)
        self.assertIn("results/unified_rules_3000/semantic_experience_generalized_for_cluster.json", rebuild)
        self.assertIn("results/unified_rules_3000/cluster_blueprints_generated.json", rebuild)

        structure_validation = build_catalog_structure_validation_command(dataset="3000")
        self.assertIn("scripts/validate_unified_catalog_structure.py", structure_validation)
        self.assertIn("results/unified_rules_3000/catalog_structure_validation.json", structure_validation)
        self.assertIn("--fail-on-invalid", structure_validation)

    def test_build_blueprints_subcommand_uses_canonical_proposal_output(self) -> None:
        root = _case_dir(self)
        paths = dataset_paths("mini", root=root)
        paths["result_dir"].mkdir(parents=True, exist_ok=True)
        paths["generated_blueprints"].parent.mkdir(parents=True, exist_ok=True)
        paths["catalog"].parent.mkdir(parents=True, exist_ok=True)
        paths["catalog"].write_text(
            json.dumps(
                {
                    "domains": [
                        {
                            "name": "Mechanics",
                            "topics": [
                                {
                                    "name": "Kinematics",
                                    "rules": [{"rule_id": "r1"}],
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
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
        root = _case_dir(self)
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
        paths["rule_embedding_input"].write_text(
            json.dumps(
                {
                    "rules": [
                        {
                            "rule_id": f"r{index}",
                            "topic_key": "mechanics::kinematics",
                            "embedding_text": f"rule {index}",
                        }
                        for index in range(1, 5)
                    ]
                }
            ),
            encoding="utf-8",
        )

        report = run_analyze_embedding_clusters(dataset="mini", root=root)

        self.assertEqual(report["total_rule_count"], 4)
        self.assertTrue(paths["rule_embedding_cluster_report"].exists())

    def test_quality_report_subcommand_uses_canonical_outputs(self) -> None:
        root = _case_dir(self)
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
        root = _case_dir(self)
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
        root = _case_dir(self)
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

    def test_candidate_and_formal_preparation_are_separate(self) -> None:
        root = _case_dir(self)
        paths = dataset_paths("mini", root=root)
        paths["result_dir"].mkdir(parents=True, exist_ok=True)
        paths["catalog"].parent.mkdir(parents=True, exist_ok=True)
        knowledge_path = root / "knowledge.json"
        tagged_path = root / "tagged.json"
        baseline_path = root / "baseline.json"
        knowledge_path.write_text(
            json.dumps(
                {
                    "domains": [
                        {
                            "name": "Mechanics",
                            "topics": [{"name": "Kinematics", "rules": []}],
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
                    "metadata": {},
                    "domains": [
                        {
                            "name": "Mechanics",
                            "topics": [
                                {
                                    "name": "Kinematics",
                                    "rules": [
                                        {
                                            "rule_id": "base_1",
                                            "title": "Baseline",
                                            "summary": "Baseline rule",
                                            "trigger": "motion",
                                            "check_logic": "check motion",
                                            "error_type": "logic",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        candidate = {
            "rule_id": "candidate_1",
            "domain": "Mechanics",
            "topic": "Kinematics",
            "title": "Candidate",
            "trigger": "motion",
            "check_logic": "check units",
            "error_type": "calculation",
            "count": 2,
            "sample_ids": ["s1", "s2"],
        }
        paths["distilled"].write_text(
            json.dumps({"rules": [candidate]}),
            encoding="utf-8",
        )

        candidate_report = run_prepare_candidates(
            dataset="mini",
            root=root,
            knowledge_path=knowledge_path,
            tagged_path=tagged_path,
        )

        self.assertEqual(candidate_report["normalization"]["baseline_seed_rules"], 0)
        self.assertTrue(paths["candidate_catalog"].exists())
        self.assertTrue(paths["rule_embedding_input"].exists())

        generalized = dict(candidate, rule_id="gen_1", title="Generalized")
        paths["generalized"].write_text(
            json.dumps(
                {
                    "metadata": {
                        "generator": "experience_candidate_generalizer_v1",
                        "scope_mode": "full",
                        "complete": True,
                        "min_source_candidates": 2,
                        "min_source_samples": 2,
                    },
                    "rules": [generalized],
                    "cluster_results": [
                        {
                            "input_candidate_ids": ["candidate_1", "candidate_2"],
                            "mappings": [
                                {
                                    "rule_id": "gen_1",
                                    "source_candidate_ids": ["candidate_1", "candidate_2"],
                                }
                            ]
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        formal_report = run_prepare_generalized(
            dataset="mini",
            root=root,
            knowledge_path=knowledge_path,
            tagged_path=tagged_path,
            baseline_catalog_path=baseline_path,
            scenario_cluster_blueprints_paths=[],
        )

        formal_payload = json.loads(paths["generalized_for_cluster"].read_text(encoding="utf-8"))
        formal_ids = {rule["rule_id"] for rule in formal_payload["rules"]}
        self.assertEqual(formal_report["normalization"]["baseline_seed_rules"], 1)
        self.assertEqual(formal_ids, {"base_1", "gen_1"})
        self.assertEqual(
            formal_payload["metadata"]["generator"],
            "experience_candidate_generalizer_v1",
        )
        self.assertTrue(formal_payload["metadata"]["formal_support_validated"])
        self.assertTrue(paths["formal_rule_embedding_input"].exists())


if __name__ == "__main__":
    unittest.main()
