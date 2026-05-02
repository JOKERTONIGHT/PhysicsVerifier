from __future__ import annotations

import unittest

from core.rule_catalog_retrieval import topic_rule_leaves
from rule_framework.builder import build_unified_catalog_from_data
from rule_framework.maintenance import add_experience_rules, attach_symbolic_bindings, recluster_catalog, remove_rules
from rule_framework.validation import validate_catalog


def _knowledge() -> dict:
    return {
        "domains": [
            {
                "name": "Mechanics",
                "topics": [
                    {
                        "name": "Kinematics",
                        "rules": [
                            {
                                "id": "kin_ref",
                                "title": "Reference kinematics",
                                "description": "v = u + at",
                                "check_logic": "Check velocity relation",
                            }
                        ],
                    }
                ],
            }
        ]
    }


def _distilled() -> dict:
    return {
        "rules": [
            {
                "rule_id": "exp_kin_1",
                "domain": "Mechanics",
                "topic": "Kinematics",
                "title": "速度方向一致性",
                "trigger": "出现 velocity 和 displacement",
                "check_logic": "速度方向应与位移符号一致",
                "error_type": "logic",
                "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["v", "x"]},
                "count": 2,
                "sample_ids": ["a", "b"],
            }
        ]
    }


class RuleFrameworkTests(unittest.TestCase):
    def test_build_catalog_adds_tree_and_rule_paths(self) -> None:
        catalog = build_unified_catalog_from_data(_knowledge(), _distilled(), [])
        topic = catalog["domains"][0]["topics"][0]
        leaves = topic_rule_leaves(topic)

        self.assertEqual(catalog["metadata"]["catalog_type"], "unified_rules_v2")
        self.assertEqual(catalog["metadata"]["hierarchy"], ["domain", "topic", "context", "cluster", "rule"])
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0]["path"]["domain"], "Mechanics")
        self.assertEqual(leaves[0]["path"]["topic"], "Kinematics")
        self.assertIn("rule_tree", topic)
        self.assertTrue(validate_catalog(catalog).ok)

    def test_add_remove_recluster_and_bind_symbolic(self) -> None:
        catalog = build_unified_catalog_from_data(_knowledge(), _distilled(), [])
        new_rule = {
            "rule_id": "exp_kin_2",
            "domain": "Mechanics",
            "topic": "Kinematics",
            "title": "加速度符号一致性",
            "trigger": "出现 acceleration 和 velocity",
            "check_logic": "加速度符号应符合速度变化方向",
            "error_type": "logic",
            "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["a", "v"]},
            "count": 1,
            "sample_ids": ["c"],
        }

        added = add_experience_rules(catalog, [new_rule])
        self.assertEqual(added.changed_rule_ids, ["exp_kin_2"])
        self.assertTrue(validate_catalog(added.catalog).ok)

        duplicate = add_experience_rules(added.catalog, [new_rule])
        self.assertEqual(duplicate.changed_rule_ids, [])
        self.assertTrue(duplicate.warnings)

        bound = attach_symbolic_bindings(
            added.catalog,
            {"checks": [{"rule_id": "exp_kin_2", "function_name": "check_exp_kin_2"}]},
        )
        leaf = [rule for rule in topic_rule_leaves(bound.catalog["domains"][0]["topics"][0]) if rule["rule_id"] == "exp_kin_2"][0]
        self.assertEqual(leaf["symbolic_binding"]["function_name"], "check_exp_kin_2")

        reclustered = recluster_catalog(bound.catalog, domain="Mechanics", topic="Kinematics")
        self.assertGreaterEqual(len(reclustered.changed_rule_ids), 2)

        removed = remove_rules(reclustered.catalog, ["exp_kin_1"])
        remaining_ids = [rule["rule_id"] for rule in topic_rule_leaves(removed.catalog["domains"][0]["topics"][0])]
        self.assertEqual(remaining_ids, ["exp_kin_2"])
        self.assertTrue(validate_catalog(removed.catalog).ok)


if __name__ == "__main__":
    unittest.main()
