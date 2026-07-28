from __future__ import annotations

import unittest

from scripts.validate_unified_catalog_structure import validate_catalog_structure


def _catalog() -> dict:
    return {
        "metadata": {
            "total_domains": 1,
            "total_topics": 1,
            "topics_with_rules": 1,
            "total_executable_rules": 2,
            "total_scenario_clusters": 1,
        },
        "domains": [
            {
                "id": "mechanics",
                "name": "Mechanics",
                "topics": [
                    {
                        "id": "mechanics.kinematics",
                        "name": "Kinematics",
                        "rules": [{"rule_id": "r1"}, {"rule_id": "r2"}],
                        "scenario_clusters": [
                            {
                                "id": "motion",
                                "rule_ids": ["r1", "r2"],
                                "rule_groups": [
                                    {"id": "checks", "rule_ids": ["r1", "r2"]}
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
    }


class UnifiedCatalogStructureValidationTests(unittest.TestCase):
    def test_valid_catalog_has_complete_rule_reachability(self) -> None:
        report = validate_catalog_structure(_catalog())

        self.assertTrue(report["valid"])
        self.assertEqual(report["counts"]["rules"], 2)

    def test_unreachable_duplicate_and_unknown_rules_are_rejected(self) -> None:
        catalog = _catalog()
        cluster = catalog["domains"][0]["topics"][0]["scenario_clusters"][0]
        cluster["rule_ids"] = ["r1", "r1", "unknown"]
        cluster["rule_groups"][0]["rule_ids"] = ["r1"]

        report = validate_catalog_structure(catalog)

        self.assertFalse(report["valid"])
        self.assertTrue(report["errors"]["topics_with_unknown_cluster_rules"])
        self.assertTrue(report["errors"]["topics_with_duplicate_cluster_assignments"])
        self.assertTrue(report["errors"]["topics_with_unreachable_rules"])
        self.assertTrue(report["errors"]["clusters_with_invalid_groups"])

    def test_global_duplicate_rule_ids_and_metadata_drift_are_rejected(self) -> None:
        catalog = _catalog()
        duplicate_topic = {
            "name": "Dynamics",
            "rules": [{"rule_id": "r1"}],
            "scenario_clusters": [{"id": "general", "rule_ids": ["r1"]}],
        }
        catalog["domains"][0]["topics"].append(duplicate_topic)

        report = validate_catalog_structure(catalog)

        self.assertFalse(report["valid"])
        self.assertEqual(report["errors"]["duplicate_global_rule_ids"], ["r1"])
        self.assertIn("total_topics", report["errors"]["metadata_mismatches"])
        self.assertIn("total_executable_rules", report["errors"]["metadata_mismatches"])


if __name__ == "__main__":
    unittest.main()
