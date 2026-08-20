import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_incremental_update import prepare_incremental_update


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(rule_id: str, title: str, sample_id: str) -> dict:
    return {
        "rule_id": rule_id,
        "domain": "Mechanics",
        "topic": "Kinematics",
        "title": title,
        "trigger": f"Trigger {title}",
        "check_logic": f"Check {title}",
        "error_type": "logic",
        "sample_ids": [sample_id],
        "count": 1,
    }


class PrepareIncrementalUpdateTests(unittest.TestCase):
    def test_prepares_isolated_workspace_and_cache_reusing_runbook(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {
                "current_candidates": root / "current_candidates.json",
                "new": root / "new.json",
                "generalized": root / "generalized.json",
                "formal": root / "formal.json",
                "proposals": root / "proposals.json",
                "catalog": root / "catalog.json",
            }
            _write(paths["current_candidates"], {"rules": [_candidate("r1", "One", "s1")]})
            _write(paths["new"], {"rules": [_candidate("r2", "Two", "s2")]})
            _write(paths["generalized"], {"rules": [], "cluster_results": []})
            _write(paths["formal"], {"rules": []})
            _write(paths["proposals"], {"proposals": []})
            _write(paths["catalog"], {"domains": []})
            workspace = root / "workspace"
            candidate_cache = root / "candidate_cache.json"
            formal_cache = root / "formal_cache.json"

            with patch(
                "scripts.prepare_incremental_update.prepare_rules_for_cluster"
            ) as prepare:
                manifest = prepare_incremental_update(
                    new_candidates_path=paths["new"],
                    workspace=workspace,
                    current_candidates_path=paths["current_candidates"],
                    current_generalized_path=paths["generalized"],
                    current_formal_path=paths["formal"],
                    current_cluster_proposals_path=paths["proposals"],
                    current_catalog_path=paths["catalog"],
                    knowledge_path=root / "knowledge.json",
                    tagged_path=root / "tagged.json",
                    baseline_catalog_path=root / "baseline.json",
                    candidate_embedding_cache_path=candidate_cache,
                    formal_embedding_cache_path=formal_cache,
                )

            prepare.assert_called_once()
            self.assertEqual(manifest["status"], "prepared")
            self.assertEqual(
                manifest["affected_topics"],
                [{"domain": "Mechanics", "topic": "Kinematics"}],
            )
            self.assertEqual(len(manifest["commands"]), 6)
            self.assertIn(
                str(candidate_cache).replace("\\", "/"),
                manifest["commands"][0]["command"],
            )
            self.assertEqual(
                manifest["formal_seed_catalog"],
                str(paths["catalog"]),
            )
            self.assertIn(
                str(paths["catalog"]).replace("\\", "/"),
                manifest["commands"][2]["command"],
            )
            self.assertIn(
                "--preserve-baseline-rule-ids",
                manifest["commands"][2]["command"],
            )
            self.assertTrue((workspace / "incremental_manifest.json").exists())
            self.assertTrue((workspace / "semantic_experience_generalized.json").exists())
            self.assertTrue((workspace / "cluster_proposals.json").exists())


if __name__ == "__main__":
    unittest.main()
