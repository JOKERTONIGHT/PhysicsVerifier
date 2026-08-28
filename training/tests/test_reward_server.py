from __future__ import annotations

import asyncio
import unittest

from training.reward_server import physics_reward_server as server


class _FakeVerifier:
    def __init__(self) -> None:
        self.calls = 0

    def verify(self, sample):
        self.calls += 1
        pred = str(sample.get("prediction") or "")
        loc = 0 if not pred else min(12, max(0, len(pred) // 3))
        return {
            "diagnostics": [
                {
                    "severity": "error",
                    "rule": "test_rule",
                    "message": "test error",
                    "start_char": loc,
                    "end_char": loc + 4,
                    "location": {"start_char": loc, "end_char": loc + 4, "paragraph_index": 1},
                }
            ]
        }


class RewardServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_mode = server.REWARD_MODE
        self.original_on_wrong = server.VERIFIER_ON_WRONG
        self.original_get_verifier = server._get_verifier
        self.original_append_metrics = server._append_metrics
        server._append_metrics = lambda record: None
        server.reset_reward_cache(maxsize=64)
        self.original_llm_judge = server._get_llm_step_judge

    def tearDown(self) -> None:
        server.REWARD_MODE = self.original_mode
        server.VERIFIER_ON_WRONG = self.original_on_wrong
        server._get_verifier = self.original_get_verifier
        server._append_metrics = self.original_append_metrics
        server._get_llm_step_judge = self.original_llm_judge

    def test_wrong_answer_skips_verifier(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "answer_low_verifier"
        server.VERIFIER_ON_WRONG = False
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(prompt="question", response=r"\boxed{2}", label="1")
            )
        )
        self.assertFalse(result["acc"])
        self.assertEqual(result["verifier_mode"], "skipped")
        self.assertEqual(fake.calls, 0)

    def test_process_paragraph_runs_verifier_on_wrong_answer(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "process_paragraph"
        server.VERIFIER_ON_WRONG = True
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(
                    prompt="question",
                    response="A derivation that is locally wrong but has no boxed match.",
                    label="1",
                )
            )
        )
        self.assertFalse(result["acc"])
        self.assertEqual(result["verifier_mode"], "full")
        self.assertEqual(fake.calls, 1)
        self.assertGreater(float(result["score"]), 0.0)

    def test_process_paragraph_ignores_final_answer(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "process_paragraph"
        server.VERIFIER_ON_WRONG = True
        server._get_verifier = lambda: fake
        text = (
            "A derivation that uses F = ma and conservation of energy. " * 4
            + r" \boxed{42}"
        )
        wrong = asyncio.run(
            server.score_one(server.ScoreRequest(prompt="question", response=text, label="1"))
        )
        right = asyncio.run(
            server.score_one(server.ScoreRequest(prompt="question", response=text, label="42"))
        )
        self.assertFalse(wrong["acc"])
        self.assertTrue(right["acc"])
        self.assertEqual(wrong["score"], right["score"])
        self.assertEqual(wrong["reward_components"]["weights"]["answer"], 0.0)
        self.assertEqual(wrong["reward_components"]["weights"]["format"], 0.0)
        self.assertEqual(fake.calls, 2)

    def test_reward_cache_dedupes_identical_completions(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "process_paragraph"
        server.VERIFIER_ON_WRONG = True
        server._get_verifier = lambda: fake
        text = "A derivation that uses F = ma and conservation of energy. " * 4
        req = server.OpenRLHFRewardRequest(
            query=[text, text, text + " extra"],
            prompts=["q", "q", "q"],
            labels=["1", "1", "1"],
        )
        payload = asyncio.run(server.openrlhf_get_reward(req))
        self.assertEqual(len(payload["rewards"]), 3)
        self.assertEqual(payload["rewards"][0], payload["rewards"][1])
        self.assertEqual(fake.calls, 2)
        self.assertEqual(payload["extra_logs"]["physics_reward_batch_unique_scored"], 2.0)

    def test_group_indices_by_key_preserves_order(self) -> None:
        groups = server.group_indices_by_key(["a", "b", "a", "c", "b"])
        self.assertEqual(groups, [[0, 2], [1, 4], [3]])

    def test_correct_answer_calls_verifier(self) -> None:
        fake = _FakeVerifier()
        server.REWARD_MODE = "answer_low_verifier"
        server._get_verifier = lambda: fake
        result = asyncio.run(
            server.score_one(
                server.ScoreRequest(prompt="question", response=r"\boxed{1}", label="1")
            )
        )
        self.assertTrue(result["acc"])
        self.assertEqual(result["verifier_mode"], "full")
        self.assertEqual(result["n_errors"], 1)
        self.assertEqual(fake.calls, 1)


class _FakeLLMJudge:
    prompt_version = "llm_step_v1"

    def __init__(self) -> None:
        self.calls: list = []

    def score_group(self, question, solutions):
        self.calls.append((question, tuple(solutions)))
        out = []
        for i, _sol in enumerate(solutions):
            out.append(
                {
                    "id": f"c{i}",
                    "raw_score": 5.0 + i,
                    "score": (5.0 + i) / 10.0,
                    "fatal_error": False,
                    "answer_only": False,
                    "step_assessments": [],
                    "brief_reason": "ok",
                }
            )
        return out

    def metrics_snapshot(self):
        return {"llm_step_api_calls": float(len(self.calls))}


class LLMStepRewardServerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_mode = server.REWARD_MODE
        self.original_get_verifier = server._get_verifier
        self.original_judge = server._get_llm_step_judge
        server._append_metrics = lambda record: None
        server.reset_reward_cache(maxsize=64)

    def tearDown(self) -> None:
        server.REWARD_MODE = self.original_mode
        server._get_verifier = self.original_get_verifier
        server._get_llm_step_judge = self.original_judge

    def test_labels_do_not_change_cache_or_reward(self) -> None:
        fake = _FakeLLMJudge()
        server.REWARD_MODE = "llm_step_score"
        server._get_llm_step_judge = lambda: fake
        a = asyncio.run(
            server.openrlhf_get_reward(
                server.OpenRLHFRewardRequest(query=["sol"], prompts=["q"], labels=["GOLD1"])
            )
        )
        b = asyncio.run(
            server.openrlhf_get_reward(
                server.OpenRLHFRewardRequest(query=["sol"], prompts=["q"], labels=["GOLD2"])
            )
        )
        self.assertEqual(a["rewards"], b["rewards"])
        self.assertEqual(len(fake.calls), 1)

    def test_same_question_group_is_one_call_and_order_restored(self) -> None:
        fake = _FakeLLMJudge()
        server.REWARD_MODE = "llm_step_score"
        server._get_llm_step_judge = lambda: fake
        payload = asyncio.run(
            server.openrlhf_get_reward(
                server.OpenRLHFRewardRequest(
                    query=["s0", "t0", "s1"],
                    prompts=["qa", "qb", "qa"],
                    labels=["1", "2", "3"],
                )
            )
        )
        self.assertEqual(len(fake.calls), 2)
        self.assertAlmostEqual(payload["rewards"][0], 0.5)
        self.assertAlmostEqual(payload["rewards"][2], 0.6)
        self.assertEqual(len(payload["rewards"]), 3)

    def test_llm_mode_never_instantiates_rule_verifier(self) -> None:
        fake = _FakeLLMJudge()
        server.REWARD_MODE = "llm_step_score"
        server._get_llm_step_judge = lambda: fake

        def boom():
            raise AssertionError("rule verifier should not be created")

        server._get_verifier = boom
        payload = asyncio.run(
            server.openrlhf_get_reward(
                server.OpenRLHFRewardRequest(query=["s0", "s1"], prompts=["q", "q"], labels=["x", "y"])
            )
        )
        self.assertEqual(payload["extra_logs"]["physics_llm_step_mode"], 1.0)
        self.assertEqual(len(fake.calls), 1)


if __name__ == "__main__":
    unittest.main()
