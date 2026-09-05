from __future__ import annotations

import unittest

from training.rl_data.screen_training_data import (
    gold_fit_drop_reason,
    looks_concatenated,
    prompt_drop_reason,
    visual_drop_reason,
)


class VisualScreenTests(unittest.TestCase):
    def test_drops_explicit_figure(self) -> None:
        self.assertEqual(visual_drop_reason("See Figure 2 for the circuit."), "visual_input")

    def test_drops_shown_below(self) -> None:
        self.assertEqual(visual_drop_reason("The side view is shown below."), "visual_input")

    def test_keeps_significant_figures(self) -> None:
        self.assertIsNone(visual_drop_reason("Report the answer to three significant figures."))

    def test_drops_separate_page_map(self) -> None:
        self.assertEqual(
            visual_drop_reason("The map shown on a separate page shows isobars."),
            "visual_input",
        )


class ConcatAndGoldFitTests(unittest.TestCase):
    def test_concat_requires_two_question_headers(self) -> None:
        self.assertFalse(looks_concatenated("1. A mass m hangs from a spring. Find T."))
        long_q = "Question: " + ("A" * 90) + " Question: second unrelated problem"
        self.assertTrue(looks_concatenated(long_q))

    def test_gold_fit_phrases(self) -> None:
        self.assertEqual(gold_fit_drop_reason("We accept reference result here."), "gold_fit")
        self.assertIsNone(
            gold_fit_drop_reason("The derived period is 2 pi sqrt(m/k). This matches the reference.")
        )


class PromptDropTests(unittest.TestCase):
    def test_known_unusable_id(self) -> None:
        self.assertEqual(prompt_drop_reason({"sample_id": "132476", "question": "ok text"}), "known_unusable")

    def test_keeps_complete_text_stem(self) -> None:
        row = {
            "sample_id": "33658",
            "question": "A mass m is attached to a spring of constant k. Find the times when F may be removed.",
        }
        self.assertIsNone(prompt_drop_reason(row))


if __name__ == "__main__":
    unittest.main()
