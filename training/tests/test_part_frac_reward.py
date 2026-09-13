from __future__ import annotations

import asyncio
import json
import statistics
import unittest
from pathlib import Path

from training.compat.part_scoring import score_prediction
from training.reward_server import physics_reward_server as server
from training.reward_server.paragraph_process import group_rank_normalize, rank_normalize_grouped

ROOT = Path(__file__).resolve().parents[2]
RFT_PATH = ROOT / "data/rl/rft_solutions_dedup.jsonl"


def _assistant_text(row: dict) -> str:
    for msg in reversed(row.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return str(row.get("solution") or "")


class PartFracRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_mode = server.REWARD_MODE
        self.original_w_answer = server.W_ANSWER
        self.original_w_format = server.W_FORMAT
        self.original_w_process = server.W_PROCESS
        self.original_get_verifier = server._get_verifier
        self.original_judge = server._get_llm_step_judge
        server._append_metrics = lambda record: None
        server.reset_reward_cache(maxsize=64)
        server.REWARD_MODE = "outcome_only"
        server.W_ANSWER = 1.0
        server.W_FORMAT = 0.05
        server.W_PROCESS = 0.0

    def tearDown(self) -> None:
        server.REWARD_MODE = self.original_mode
        server.W_ANSWER = self.original_w_answer
        server.W_FORMAT = self.original_w_format
        server.W_PROCESS = self.original_w_process
        server._get_verifier = self.original_get_verifier
        server._get_llm_step_judge = self.original_judge

    def test_two_part_gold_is_partial_not_any_of(self) -> None:
        def boom():
            raise AssertionError("verifier should not run")

        server._get_verifier = boom
        payload = asyncio.run(
            server.openrlhf_get_reward(
                server.OpenRLHFRewardRequest(
                    query=[r"work \boxed{1.16} and \boxed{wrong}", r"work \boxed{1.16} and \boxed{3.07e7}"],
                    prompts=["q", "q"],
                    labels=[["1.16", "3.07e7"], ["1.16", "3.07e7"]],
                )
            )
        )
        half, full = payload["rewards"]
        self.assertAlmostEqual(half, 0.5 + 0.05, places=5)
        self.assertAlmostEqual(full, 1.0 + 0.05, places=5)
        self.assertGreater(full, half)

    def test_rft_solutions_median_part_frac(self) -> None:
        if not RFT_PATH.is_file():
            self.skipTest(f"missing {RFT_PATH}")
        rows = []
        with RFT_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
                if len(rows) >= 40:
                    break
        self.assertGreaterEqual(len(rows), 8)
        own = []
        shuffled = []
        golds = [r.get("solution") for r in rows]
        for i, row in enumerate(rows):
            pred = _assistant_text(row)
            labels = [str(row.get("solution") or "")]
            own.append(float(score_prediction(pred, labels)["part_frac"]))
            other = golds[(i + 7) % len(golds)]
            shuffled.append(float(score_prediction(pred, [str(other or "xxx")])["part_frac"]))
        own_med = statistics.median(own)
        shuf_med = statistics.median(shuffled)
        self.assertGreaterEqual(
            own_med,
            0.9,
            f"r_answer median on RFT solutions is {own_med:.3f}; reward/data mismatch, do not train",
        )
        self.assertLessEqual(shuf_med, 0.1, f"shuffled median {shuf_med:.3f} should be ~0")


class RankNormalizeTests(unittest.TestCase):
    def test_order_and_ties(self) -> None:
        ranked = group_rank_normalize([0.1, 0.9, 0.1])
        self.assertAlmostEqual(ranked[1], 1.0)
        self.assertAlmostEqual(ranked[0], ranked[2])
        grouped = rank_normalize_grouped(["a", "a", "b"], [0.2, 0.8, 0.5])
        self.assertAlmostEqual(grouped[0], 0.0)
        self.assertAlmostEqual(grouped[1], 1.0)
        self.assertAlmostEqual(grouped[2], 0.5)


if __name__ == "__main__":
    unittest.main()
