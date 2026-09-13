#!/usr/bin/env python3
from __future__ import annotations

import unittest

from training.swift.eval_verifier_auc import average_precision, point_biserial, roc_auc


class VerifierAucTests(unittest.TestCase):
    def test_perfect_ranker(self) -> None:
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        self.assertGreaterEqual(roc_auc(scores, labels), 0.99)
        self.assertGreater(average_precision(scores, labels), 0.9)
        self.assertGreater(point_biserial(scores, labels), 0.5)

    def test_inverted_ranker_below_gate(self) -> None:
        scores = [0.9, 0.8, 0.2, 0.1]
        labels = [0, 0, 1, 1]
        self.assertLess(roc_auc(scores, labels), 0.35)


if __name__ == "__main__":
    unittest.main()
