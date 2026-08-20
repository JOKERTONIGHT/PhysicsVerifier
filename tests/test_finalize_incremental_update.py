import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.finalize_incremental_update import finalize_incremental_update


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _catalog(rule_ids: list[str]) -> dict:
    return {
        "domains": [
            {
                "name": "Mechanics",
                "topics": [
                    {
                        "name": "Kinematics",
                        "rules": [{"rule_id": rule_id} for rule_id in rule_ids],
                        "scenario_clusters": [
                            {"id": "coarse", "rule_ids": rule_ids}
                        ],
                    }
                ],
            }
        ]
    }


class FinalizeIncrementalUpdateTests(unittest.TestCase):
    def test_removed_base_rule_blocks_retrieval_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            _write(
                workspace / "incremental_manifest.json",
                {
                    "affected_topics": [
                        {"domain": "Mechanics", "topic": "Kinematics"}
                    ]
                },
            )
            for name in (
                "catalog_precluster.json",
                "cluster_proposals.json",
                "semantic_experience_distilled_for_cluster.json",
                "semantic_experience_generalized.json",
                "semantic_experience_generalized_for_cluster.json",
            ):
                _write(workspace / name, {"rules": [], "proposals": []})
            base_path = root / "base.json"
            _write(base_path, _catalog(["old"]))

            with (
                patch(
                    "scripts.finalize_incremental_update.add_catalog_fallback_proposals",
                    side_effect=lambda proposals, _catalog: proposals,
                ),
                patch(
                    "scripts.finalize_incremental_update.build_generated_blueprints_from_refined_proposals",
                    return_value={},
                ),
                patch(
                    "scripts.finalize_incremental_update.build_unified_catalog",
                    return_value=_catalog(["new"]),
                ),
                patch(
                    "scripts.finalize_incremental_update.validate_catalog_structure",
                    return_value={"valid": True},
                ),
                patch(
                    "scripts.finalize_incremental_update.audit_rule_coarsening",
                    return_value={"complete": True},
                ),
            ):
                report = finalize_incremental_update(
                    workspace=workspace,
                    base_catalog_path=base_path,
                )

            self.assertFalse(report["ready_for_retrieval_evaluation"])
            self.assertFalse(report["promotion_ready"])
            self.assertEqual(
                report["change_scope"]["unexpected_changed_topics"],
                [],
            )
            self.assertEqual(report["change_scope"]["added_rule_ids"], ["new"])
            self.assertEqual(report["change_scope"]["removed_rule_ids"], ["old"])
            self.assertFalse(report["change_scope"]["identity_stable"])

    def test_preserves_base_ids_and_allows_additive_catalog_for_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            workspace = root / "workspace"
            workspace.mkdir()
            _write(
                workspace / "incremental_manifest.json",
                {
                    "affected_topics": [
                        {"domain": "Mechanics", "topic": "Kinematics"}
                    ]
                },
            )
            for name in (
                "catalog_precluster.json",
                "cluster_proposals.json",
                "semantic_experience_distilled_for_cluster.json",
                "semantic_experience_generalized.json",
                "semantic_experience_generalized_for_cluster.json",
            ):
                _write(workspace / name, {"rules": [], "proposals": []})
            base_path = root / "base.json"
            _write(base_path, _catalog(["old"]))

            with (
                patch(
                    "scripts.finalize_incremental_update.add_catalog_fallback_proposals",
                    side_effect=lambda proposals, _catalog: proposals,
                ),
                patch(
                    "scripts.finalize_incremental_update.build_generated_blueprints_from_refined_proposals",
                    return_value={},
                ),
                patch(
                    "scripts.finalize_incremental_update.build_unified_catalog",
                    return_value=_catalog(["old", "new"]),
                ),
                patch(
                    "scripts.finalize_incremental_update.validate_catalog_structure",
                    return_value={"valid": True},
                ),
                patch(
                    "scripts.finalize_incremental_update.audit_rule_coarsening",
                    return_value={"complete": True},
                ),
            ):
                report = finalize_incremental_update(
                    workspace=workspace,
                    base_catalog_path=base_path,
                )

            self.assertTrue(report["ready_for_retrieval_evaluation"])
            self.assertTrue(report["change_scope"]["identity_stable"])
            self.assertEqual(report["change_scope"]["added_rule_ids"], ["new"])
            self.assertEqual(report["change_scope"]["removed_rule_ids"], [])


if __name__ == "__main__":
    unittest.main()
