from __future__ import annotations

import unittest

from training.swift.monitor_process_reward import evaluate
from training.swift.report_group_gradient import summarize


class ReportGroupGradientTests(unittest.TestCase):
    def test_mixed_and_gates(self) -> None:
        rows = []
        for i in range(8):
            rows.append(
                {
                    "sample_id": "mix",
                    "acc": i < 3,
                    "response": r"reason. \boxed{1}",
                    "no_boxed": False,
                    "truncated": False,
                    "repetitive": False,
                }
            )
        for i in range(8):
            rows.append(
                {
                    "sample_id": "wrong",
                    "acc": False,
                    "response": r"reason. \boxed{0}",
                    "no_boxed": False,
                    "truncated": False,
                    "repetitive": False,
                }
            )
        report = summarize(rows, min_mixed=0.4, max_trunc=0.10, min_boxed=0.95)
        self.assertAlmostEqual(report["mixed_rate"], 0.5)
        self.assertTrue(report["pass"])
        self.assertGreaterEqual(report["boxed_rate_complete"], 0.95)
        fail = summarize(rows, min_mixed=0.9, max_trunc=0.10, min_boxed=0.95)
        self.assertFalse(fail["pass"])


class SummarizeRftDiagTests(unittest.TestCase):
    def test_finds_first_no_boxed_jump(self) -> None:
        from training.swift.summarize_rft_diag import summarize_gate

        report = {
            "checkpoints": [
                {
                    "ckpt": "/x/checkpoint-20",
                    "sft_part_avg_at_k": 0.25,
                    "sft_degrade_rate": 0.02,
                    "sft_no_boxed_rate": 0.002,
                    "pass": True,
                },
                {
                    "ckpt": "/x/checkpoint-40",
                    "sft_part_avg_at_k": 0.18,
                    "sft_degrade_rate": 0.12,
                    "sft_no_boxed_rate": 0.23,
                    "pass": False,
                },
            ]
        }
        collapse = summarize_gate(report)
        self.assertEqual(collapse["first_no_boxed_ge_5pct_step"], 40)
        self.assertEqual(collapse["first_degrade_ge_5pct_step"], 40)
        self.assertAlmostEqual(collapse["max_no_boxed_rate"], 0.23)


class MonitorProcessRewardTests(unittest.TestCase):
    def test_zero_std_three_steps_stops(self) -> None:
        metrics = [{"physics_reward_group_std_mean": 0.0} for _ in range(3)]
        report = evaluate(metrics, [{"loss": 0.1, "reward": 0.2}])
        self.assertTrue(report["stop"])
        self.assertIn("zero_std_three_steps", report["reasons"])

    def test_acc_decline_stops(self) -> None:
        metrics = [{"physics_answer_acc": 0.30 - 0.01 * i} for i in range(12)]
        report = evaluate(metrics, [{"loss": 0.1, "reward": 0.4}] * 4)
        self.assertTrue(report["stop"])
        self.assertIn("answer_acc_declining", report["reasons"])

    def test_length_explosion_stops(self) -> None:
        train = [{"completion_length": 100.0, "reward": 0.2} for _ in range(4)]
        train += [{"completion_length": 200.0, "reward": 0.4} for _ in range(4)]
        report = evaluate([{"physics_reward_group_std_mean": 0.2}], train)
        self.assertTrue(report["stop"])
        self.assertIn("reward_up_length_explosion", report["reasons"])

    def test_low_mixed_warns_only(self) -> None:
        report = evaluate(
            [{"physics_mixed_acc_group_rate": 0.05, "physics_reward_group_std_mean": 0.2}],
            [{"loss": 0.1, "reward": 0.2}],
        )
        self.assertFalse(report["stop"])
        self.assertIn("low_mixed_group_rate", report["warnings"])

    def test_truncation_rate_rising_warns(self) -> None:
        metrics = [
            {"physics_trunc_rate": 0.04, "physics_reward_group_std_mean": 0.2}
            for _ in range(3)
        ] + [
            {"physics_trunc_rate": 0.12, "physics_reward_group_std_mean": 0.2}
            for _ in range(3)
        ]
        report = evaluate(metrics, [{"loss": 0.1, "reward": 0.2}])
        self.assertFalse(report["stop"])
        self.assertIn("truncation_rate_rising", report["warnings"])


if __name__ == "__main__":
    unittest.main()
