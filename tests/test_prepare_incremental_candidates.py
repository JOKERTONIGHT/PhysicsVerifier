import unittest

from scripts.prepare_incremental_candidates import prepare_incremental_candidates


def _rule(
    rule_id: str,
    title: str,
    *,
    sample_ids: list[str] | None = None,
    trigger: str = "shared trigger",
) -> dict:
    return {
        "rule_id": rule_id,
        "domain": "Mechanics",
        "topic": "Kinematics",
        "title": title,
        "trigger": trigger,
        "check_logic": "Check the shared mechanism.",
        "error_type": "logic",
        "count": len(sample_ids or []),
        "sample_ids": sample_ids or [],
        "source_rule_ids": [],
    }


class PrepareIncrementalCandidatesTests(unittest.TestCase):
    def test_adds_unique_candidate_and_reports_affected_topic(self) -> None:
        merged, report = prepare_incremental_candidates(
            current_payload={"rules": [_rule("exp_old", "Old rule")]},
            new_payload={"rules": [_rule("exp_new", "New rule", sample_ids=["s2"])]},
        )

        self.assertEqual([rule["rule_id"] for rule in merged["rules"]], ["exp_old", "exp_new"])
        self.assertEqual(report["added_candidate_ids"], ["exp_new"])
        self.assertEqual(
            report["affected_topics"],
            [{"domain": "Mechanics", "topic": "Kinematics"}],
        )

    def test_exact_duplicate_merges_source_support_without_new_rule(self) -> None:
        current = _rule("exp_old", "Same rule", sample_ids=["s1"])
        duplicate = _rule("exp_other", "Same rule", sample_ids=["s2"])

        merged, report = prepare_incremental_candidates(
            current_payload={"rules": [current]},
            new_payload={"rules": [duplicate]},
        )

        self.assertEqual(len(merged["rules"]), 1)
        self.assertEqual(merged["rules"][0]["sample_ids"], ["s1", "s2"])
        self.assertEqual(merged["rules"][0]["count"], 2)
        self.assertEqual(report["support_updated_candidate_ids"], ["exp_old"])

    def test_exact_formal_duplicate_is_not_added(self) -> None:
        formal = _rule("gen_formal", "Covered rule", sample_ids=["s0"])
        candidate = _rule("exp_new", "Covered rule", sample_ids=["s1"])

        merged, report = prepare_incremental_candidates(
            current_payload={"rules": []},
            new_payload={"rules": [candidate]},
            formal_payload={"rules": [formal]},
        )

        self.assertEqual(merged["rules"], [])
        self.assertEqual(
            report["covered_by_formal"],
            [{"candidate_rule_id": "exp_new", "formal_rule_id": "gen_formal"}],
        )

    def test_same_rule_id_with_different_content_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "rule_id conflict"):
            prepare_incremental_candidates(
                current_payload={"rules": [_rule("exp_same", "First")]},
                new_payload={"rules": [_rule("exp_same", "Second")]},
            )


if __name__ == "__main__":
    unittest.main()
