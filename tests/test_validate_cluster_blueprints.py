from __future__ import annotations

import unittest

from scripts.validate_cluster_blueprints import validate_blueprints_against_catalog


def _catalog() -> dict:
    return {
        "domains": [
            {
                "name": "Mechanics",
                "topics": [
                    {
                        "name": "Kinematics",
                        "rules": [
                            {"rule_id": "r1"},
                            {"rule_id": "r2"},
                        ],
                    }
                ],
            }
        ]
    }


class ValidateClusterBlueprintsTests(unittest.TestCase):
    def test_subset_mode_allows_uncovered_rules_because_builder_adds_fallback(self) -> None:
        blueprints = {
            "mechanics::kinematics": [
                {
                    "cluster_id": "timing",
                    "rule_groups": [{"group_id": "timing_rules", "rule_ids": ["r1"]}],
                }
            ]
        }

        report = validate_blueprints_against_catalog(_catalog(), blueprints, mode="subset")

        self.assertTrue(report["valid"])
        self.assertEqual(report["topics_with_uncovered_rules"], ["mechanics::kinematics"])

    def test_full_mode_requires_complete_rule_coverage(self) -> None:
        blueprints = {
            "mechanics::kinematics": [
                {
                    "cluster_id": "timing",
                    "rule_groups": [{"group_id": "timing_rules", "rule_ids": ["r1"]}],
                }
            ]
        }

        report = validate_blueprints_against_catalog(_catalog(), blueprints, mode="full")

        self.assertFalse(report["valid"])
        self.assertEqual(report["topics_with_uncovered_rules"], ["mechanics::kinematics"])


if __name__ == "__main__":
    unittest.main()
