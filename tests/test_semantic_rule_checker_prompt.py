from __future__ import annotations

import unittest
from unittest.mock import patch

from core.semantic_rule_checker import SemanticRuleChecker


class SemanticRuleCheckerPromptTest(unittest.TestCase):
    def test_upstream_training_paragraph_overrides_remain_effective(self):
        checker = SemanticRuleChecker(llm_model=None, enable_cache=False)
        text = "x" * 500
        with patch.dict("os.environ", {}, clear=True):
            default = checker._paragraph_ranges(text)
        with patch.dict("os.environ", {"PHYSICS_REWARD_PARA_TARGET": "60", "PHYSICS_REWARD_PARA_MIN": "40", "PHYSICS_REWARD_PARA_MAX": "80"}, clear=True):
            configured = checker._paragraph_ranges(text)
        self.assertGreater(len(configured), len(default))

    def test_rule_check_prompt_includes_problem_and_rejects_method_omission_flags(self) -> None:
        checker = SemanticRuleChecker(llm_model=None, enable_cache=False)

        _, prompt = checker._get_check_prompt(
            srd="Use conservation of angular momentum when applicable.",
            raw_answer="The student solution.",
            problem_text="Find the oscillation period.",
            context_summary="{}",
            rule_id="rule_1",
        )

        self.assertIn("Find the oscillation period.", prompt)
        self.assertIn("conditional diagnostic aid", prompt)
        self.assertIn("Do NOT penalize", prompt)
        self.assertIn("alternative derivation", prompt)


if __name__ == "__main__":
    unittest.main()
