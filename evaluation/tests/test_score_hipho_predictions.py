from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.benchmarks.hipho.score_hipho_predictions import (
    answers_match,
    evaluate_gate,
    extract_all_boxed,
    looks_repetitive,
    looks_truncated,
    score_prediction,
    summarize_scores,
)


class ExtractBoxedTests(unittest.TestCase):
    def test_extracts_all_nested_boxes(self) -> None:
        src = r"first \boxed{\frac{1}{2}} then \boxed{4r}"
        self.assertEqual(extract_all_boxed(src), [r"\frac{1}{2}", "4r"])

    def test_unclosed_counts_as_truncated(self) -> None:
        self.assertTrue(looks_truncated(r"work \boxed{1"))
        self.assertFalse(looks_truncated(r"work \boxed{1}"))


class MatchAndPartTests(unittest.TestCase):
    def test_part_level_hits_first_box(self) -> None:
        pred = r"A \boxed{1.16} and B \boxed{wrong}"
        gold = ["1.16", r"3.07 \times 10^{7}"]
        m = score_prediction(pred, gold)
        self.assertEqual(m["n_parts"], 2)
        self.assertEqual(m["n_hit"], 1)
        self.assertFalse(m["item_correct"])
        self.assertAlmostEqual(m["part_frac"], 0.5)

    def test_item_correct_needs_all_parts(self) -> None:
        pred = r"\boxed{1.16} later \boxed{3.07e7}"
        gold = ["1.16", "3.07e7"]
        m = score_prediction(pred, gold)
        self.assertTrue(m["item_correct"])
        self.assertTrue(answers_match("1.16", "1.16"))

    def test_unit_equiv_fallback(self) -> None:
        pred = r"\boxed{2.0 \mathrm{~N}}"
        m = score_prediction(pred, [r"2 N"])
        self.assertTrue(m["item_correct"])


class DegenerateTests(unittest.TestCase):
    def test_repetition_window(self) -> None:
        chunk = "the uncertainty in the measurement. " * 20
        self.assertGreaterEqual(len(chunk), 200)
        text = chunk * 4
        self.assertTrue(looks_repetitive(text))
        self.assertFalse(looks_repetitive("short unique derivation \\boxed{1}"))

    def test_no_boxed_flag(self) -> None:
        m = score_prediction("lots of words without an answer", ["1"])
        self.assertTrue(m["no_boxed"])
        self.assertFalse(m["item_correct"])


class AggregateAndGateTests(unittest.TestCase):
    def test_avg_and_pass_at_k(self) -> None:
        rows = []
        for i, pred in enumerate([r"\boxed{1}", r"nope", r"\boxed{1}", r"nope"]):
            rows.append(
                {
                    "metadata": {"sample_id": "a"},
                    "label": ["1"],
                    "prediction": pred,
                    "sample_index": i,
                }
            )
        rows.append(
            {
                "metadata": {"sample_id": "b"},
                "label": ["2"],
                "prediction": "fail",
                "sample_index": 0,
            }
        )
        rows.append(
            {
                "metadata": {"sample_id": "b"},
                "label": ["2"],
                "prediction": r"\boxed{2}",
                "sample_index": 1,
            }
        )
        summary = summarize_scores(rows, k_hint=2)
        self.assertEqual(summary["n_samples"], 2)
        self.assertEqual(summary["k"], 2)
        self.assertGreater(summary["item_pass_at_k"], summary["item_avg_at_k"])
        self.assertAlmostEqual(summary["item_pass_at_k"], 1.0)

    def test_gate_non_degradation(self) -> None:
        base = {
            "part_avg_at_k": 0.10,
            "n_samples": 88,
            "degrade_rate": 0.20,
            "part_avg_at_k_se": 0.03,
        }
        ok = evaluate_gate({"part_avg_at_k": 0.09, "degrade_rate": 0.15}, base)
        self.assertTrue(ok["pass"])
        bad = evaluate_gate({"part_avg_at_k": 0.01, "degrade_rate": 0.15}, base)
        self.assertFalse(bad["pass"])
        deg = evaluate_gate({"part_avg_at_k": 0.12, "degrade_rate": 0.40}, base)
        self.assertFalse(deg["pass"])
        strict = evaluate_gate(
            {"part_avg_at_k": 0.20, "degrade_rate": 0.08, "no_boxed_rate": 0.10},
            {"part_avg_at_k": 0.18, "degrade_rate": 0.10, "n_samples": 65},
            min_part_avg=0.252,
            max_degrade_rate=0.05,
            max_no_boxed_rate=0.05,
        )
        self.assertFalse(strict["pass"])
        self.assertIn("min_part_avg", strict["extra_fail"])
        self.assertIn("max_degrade_rate", strict["extra_fail"])
        self.assertIn("max_no_boxed_rate", strict["extra_fail"])

    def test_cli_roundtrip(self) -> None:
        from evaluation.benchmarks.hipho import score_hipho_predictions as mod

        with tempfile.TemporaryDirectory() as td:
            pred_path = Path(td) / "pred.jsonl"
            out_path = Path(td) / "scores.json"
            rec = {
                "metadata": {"sample_id": "x"},
                "label": [r"\boxed{C}"],
                "prediction": r"reason \boxed{C}",
                "sample_index": 0,
            }
            pred_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
            argv = [
                "score_hipho_predictions.py",
                "--predictions",
                str(pred_path),
                "--output",
                str(out_path),
                "--no-use-verifier",
            ]
            import sys

            old = sys.argv
            sys.argv = argv
            try:
                mod.main()
            finally:
                sys.argv = old
            scores = json.loads(out_path.read_text())
            self.assertEqual(scores["n_samples"], 1)
            self.assertAlmostEqual(scores["item_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
