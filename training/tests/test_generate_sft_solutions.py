from __future__ import annotations

import unittest

from training.rl_data.generate_sft_solutions import (
    HINT_MARK,
    _make_sft_row,
    build_generation_messages,
    has_meta_talk,
    pick_shortest_correct,
    user_turn_has_hint,
)
from training.swift.smoke_self_judge import spearman


class PickShortestCorrectTests(unittest.TestCase):
    def test_picks_shortest_matching_boxed_answer(self) -> None:
        gold = r"\boxed{2}"
        long_ok = "long reasoning\n\\boxed{2}"
        short_ok = "\\boxed{2}"
        wrong = "\\boxed{3}"
        chosen = pick_shortest_correct([wrong, long_ok, short_ok], gold)
        self.assertEqual(chosen, short_ok)

    def test_none_when_all_wrong(self) -> None:
        self.assertIsNone(pick_shortest_correct(["\\boxed{9}", "no box"], r"\boxed{1}"))


class HintGoldIsolationTests(unittest.TestCase):
    def test_generation_prompt_can_include_hint(self) -> None:
        msgs = build_generation_messages("Find T.", r"\boxed{2}", hint_gold=True)
        user = msgs[-1]["content"]
        self.assertIn(HINT_MARK, user)
        self.assertIn("Find T.", user)

    def test_stored_row_rejects_hint_in_user_turn(self) -> None:
        src = {
            "sample_id": "x",
            "question": "Find T.",
            "solution": "2",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": f"Find T.\n{HINT_MARK}: 2"},
            ],
        }
        with self.assertRaises(ValueError):
            _make_sft_row(src, r"work\n\boxed{2}", hint_gold=True)

    def test_stored_row_has_clean_user_turn(self) -> None:
        src = {
            "sample_id": "x",
            "question": "Find T.",
            "solution": "2",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "Find T."},
            ],
        }
        rec = _make_sft_row(src, r"work\n\boxed{2}", hint_gold=True)
        self.assertFalse(user_turn_has_hint(rec["messages"]))
        self.assertTrue(rec["hint_gold"])
        self.assertFalse(has_meta_talk(rec["messages"][-1]["content"]))


class SpearmanTests(unittest.TestCase):
    def test_perfect_rank_correlation(self) -> None:
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]), 1.0)

    def test_inverse_rank_correlation(self) -> None:
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]), -1.0)


if __name__ == "__main__":
    unittest.main()
