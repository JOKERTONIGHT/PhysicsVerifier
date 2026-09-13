#!/usr/bin/env python3
from __future__ import annotations

import unittest
from pathlib import Path

from training.rl_data.build_physics_tiers import _tier_of, _to_prompt, convert_hipho_easy


class PhysicsTiersTests(unittest.TestCase):
    def test_tier_labels(self) -> None:
        self.assertEqual(_tier_of("High School and Below"), "easy")
        self.assertEqual(_tier_of("High School Olympiad"), "mid")
        self.assertEqual(_tier_of("Undergraduate/Postgraduate (Physics Major)"), "hard")
        self.assertEqual(_tier_of("Knowledge Recall"), "easy")
        self.assertEqual(_tier_of("Laws Application"), "mid")
        self.assertEqual(_tier_of("Math Derivation"), "hard")

    def test_hipho_fma_converts(self) -> None:
        path = Path("/slow_share/jinjianhan/workspace/benchmarks/hipho/hipho_text_only.jsonl")
        if not path.is_file():
            self.skipTest("hipho text-only missing")
        rows = convert_hipho_easy(path)
        self.assertGreater(len(rows), 0)
        self.assertTrue(any(r["tier"] == "easy" for r in rows))
        sample = rows[0]
        self.assertIn("messages", sample)
        self.assertTrue(sample["solution"])

    def test_skips_empty_answer(self) -> None:
        row = {"question": "What is F?", "answer": [""], "id": "x"}
        self.assertIsNone(_to_prompt(row, source="t", tier="easy"))

    def test_prefers_answers_over_long_solution(self) -> None:
        row = {
            "problem": "Find I.",
            "answers": r"\boxed{1/2}",
            "solution": "A very long derivation without a usable gold field.",
            "id": "ug-1",
        }
        converted = _to_prompt(row, source="UGPhysics", tier="mid")
        self.assertIsNotNone(converted)
        self.assertIn("1/2", str(converted["solution"]))
        self.assertNotIn("very long derivation", str(converted["solution"]))


if __name__ == "__main__":
    unittest.main()
