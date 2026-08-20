from __future__ import annotations

import unittest

from scripts.evaluate_physics_eval_sets import _fill_missing_pred_locations


class FillMissingPredictionLocationsTest(unittest.TestCase):
    def test_relocates_quote_when_model_offsets_point_elsewhere(self) -> None:
        answer = ("Introductory text " * 24) + "\n\nThe incorrect equation is E = mc."
        quote = "E = mc"
        findings = [
            {
                "quote": quote,
                "start_char": 0,
                "end_char": len(quote),
                "span_valid": True,
                "paragraph_index": 1,
                "paragraph_valid": True,
                "locate_method": "model_provided",
            }
        ]

        located = _fill_missing_pred_locations(findings, answer)[0]

        self.assertEqual(answer.index(quote), located["start_char"])
        self.assertEqual(answer.index(quote) + len(quote), located["end_char"])
        self.assertEqual(2, located["paragraph_index"])
        self.assertEqual("fallback_exact", located["locate_method"])

    def test_keeps_valid_model_offsets_but_rebuilds_paragraph(self) -> None:
        answer = ("First paragraph text " * 22) + "\n\nSecond paragraph has F = ma."
        quote = "F = ma"
        start = answer.index(quote)
        findings = [
            {
                "quote": quote,
                "start_char": start,
                "end_char": start + len(quote),
                "span_valid": True,
                "paragraph_index": 99,
                "paragraph_valid": True,
                "locate_method": "model_provided",
            }
        ]

        located = _fill_missing_pred_locations(findings, answer)[0]

        self.assertEqual(start, located["start_char"])
        self.assertEqual(2, located["paragraph_index"])
        self.assertEqual("model_provided", located["locate_method"])


if __name__ == "__main__":
    unittest.main()
