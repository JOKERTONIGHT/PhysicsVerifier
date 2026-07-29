import unittest

from scripts.audit_rule_coarsening import audit_rule_coarsening


class AuditRuleCoarseningTests(unittest.TestCase):
    def test_complete_when_candidates_are_accounted_and_formal_rules_are_reachable(self) -> None:
        candidates = {
            "rules": [
                {"rule_id": "c1"},
                {"rule_id": "c2"},
                {"rule_id": "c3"},
            ]
        }
        generalized = {
            "rules": [
                {
                    "rule_id": "g1",
                    "count": 2,
                    "sample_ids": ["s1", "s2"],
                }
            ],
            "cluster_results": [
                {
                    "mappings": [
                        {"rule_id": "g1", "source_candidate_ids": ["c1", "c2"]}
                    ]
                }
            ],
            "pending_candidate_ids": ["c3"],
        }
        formal = {"rules": [{"rule_id": "g1"}, {"rule_id": "baseline"}]}
        catalog = {
            "domains": [
                {
                    "topics": [
                        {
                            "rules": [{"rule_id": "g1"}, {"rule_id": "baseline"}],
                            "scenario_clusters": [
                                {
                                    "id": "coarse_cluster",
                                    "rule_ids": ["g1", "baseline"],
                                }
                            ],
                        }
                    ]
                }
            ]
        }

        report = audit_rule_coarsening(
            candidates=candidates,
            generalized=generalized,
            formal=formal,
            catalog=catalog,
        )

        self.assertTrue(report["complete"])
        self.assertEqual(report["counts"]["generated_multi_sample_rules"], 1)
        self.assertEqual(report["counts"]["preserved_baseline_rules"], 1)

    def test_rejects_single_sample_generated_rule_and_duplicate_assignment(self) -> None:
        candidates = {"rules": [{"rule_id": "c1"}, {"rule_id": "c2"}]}
        generalized = {
            "rules": [{"rule_id": "g1", "count": 1, "sample_ids": ["s1"]}],
            "cluster_results": [
                {
                    "mappings": [
                        {"rule_id": "g1", "source_candidate_ids": ["c1"]}
                    ]
                }
            ],
            "pending_candidate_ids": ["c2"],
        }
        formal = {"rules": [{"rule_id": "g1"}]}
        catalog = {
            "domains": [
                {
                    "topics": [
                        {
                            "rules": [{"rule_id": "g1"}],
                            "scenario_clusters": [
                                {"id": "a", "rule_ids": ["g1"]},
                                {"id": "b", "rule_ids": ["g1"]},
                            ],
                        }
                    ]
                }
            ]
        }

        report = audit_rule_coarsening(
            candidates=candidates,
            generalized=generalized,
            formal=formal,
            catalog=catalog,
        )

        self.assertFalse(report["complete"])
        self.assertFalse(
            report["gates"]["generated_rules_have_multi_sample_support"]
        )
        self.assertFalse(report["gates"]["each_catalog_rule_reachable_once"])


if __name__ == "__main__":
    unittest.main()
