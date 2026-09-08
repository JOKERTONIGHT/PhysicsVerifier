from __future__ import annotations

import unittest

from training.rl_data.generate_sft_solutions import (
    HINT_MARK,
    SFT_SYSTEM,
    _make_sft_row,
    build_generation_messages,
    is_acceptable_solution,
    is_disagree,
    pick_best_correct,
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

    def test_hint_prompt_includes_gold_but_sft_row_does_not(self) -> None:
        gold = r"\boxed{2}"
        msgs = build_generation_messages("What is 1+1?", gold, hint_gold=True)
        self.assertIn(HINT_MARK, msgs[-1]["content"])
        self.assertIn("2", msgs[-1]["content"])
        src = {
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "What is 1+1?"}],
            "question": "What is 1+1?",
            "solution": gold,
            "sample_id": "t1",
        }
        row = _make_sft_row(src, "Because 1+1=2.\n\\boxed{2}", hint_gold=True)
        self.assertTrue(row["hint_gold"])
        self.assertFalse(user_turn_has_hint(row["messages"]))
        self.assertEqual(row["messages"][-1]["content"], "Because 1+1=2.\n\\boxed{2}")

    def test_refuses_to_store_hint_in_user_turn(self) -> None:
        src = {
            "messages": [
                {"role": "user", "content": f"q\n{HINT_MARK}:\n\\boxed{1}"},
            ],
            "solution": r"\boxed{1}",
            "sample_id": "t2",
        }
        with self.assertRaises(ValueError):
            _make_sft_row(src, "\\boxed{1}", hint_gold=True)

    def test_acceptable_requires_boxed_and_min_chars(self) -> None:
        gold = r"\boxed{2}"
        self.assertFalse(is_acceptable_solution("2", gold, min_chars=0))
        self.assertFalse(is_acceptable_solution("\\boxed{2}", gold, min_chars=20))
        self.assertTrue(is_acceptable_solution("because addition\n\\boxed{2}", gold, min_chars=10))
        self.assertFalse(
            is_acceptable_solution("The reference answer is 2.\nBecause addition.\n\\boxed{2}", gold, min_chars=10)
        )

    def test_long_chain_target_and_length_cap(self) -> None:
        self.assertIn("3500-5000", SFT_SYSTEM)
        gold = r"\boxed{2}"
        long = ("Energy is conserved. " * 220) + "\n\\boxed{2}"
        self.assertGreater(len(long), 4000)
        self.assertLess(len(long), 8000)
        self.assertTrue(is_acceptable_solution(long, gold, min_chars=2500, require_style=True))
        too_long = ("Energy is conserved. " * 500) + "\n\\boxed{2}"
        self.assertFalse(is_acceptable_solution(too_long, gold, min_chars=2500, require_style=True))

    def test_fewshot_is_generation_only(self) -> None:
        gold = r"\boxed{2}"
        shots = [{"question": "ex q", "solution": "ex sol\n\\boxed{0}"}]
        msgs = build_generation_messages("What is 1+1?", gold, hint_gold=True, fewshot=shots)
        self.assertEqual(msgs[1]["content"], "ex q")
        self.assertEqual(msgs[2]["content"], "ex sol\n\\boxed{0}")
        self.assertIn("What is 1+1?", msgs[-1]["content"])
        src = {
            "messages": [{"role": "user", "content": "What is 1+1?"}],
            "question": "What is 1+1?",
            "solution": gold,
            "sample_id": "t1",
        }
        row = _make_sft_row(src, "Because 1+1=2.\n\\boxed{2}", hint_gold=True)
        self.assertFalse(any(m.get("content") == "ex q" for m in row["messages"]))

    def test_pick_best_falls_back_to_longest_graded_if_all_short(self) -> None:
        gold = r"\boxed{2}"
        chosen = pick_best_correct(["\\boxed{2}", "xx\\boxed{2}yy"], gold, min_chars=50)
        self.assertEqual(chosen, "xx\\boxed{2}yy")

    def test_disagree_is_not_acceptable(self) -> None:
        gold = r"\boxed{2}"
        self.assertTrue(is_disagree("Cannot derive the result.\n\\boxed{DISAGREE}"))
        self.assertFalse(is_acceptable_solution("Cannot derive the result.\n\\boxed{DISAGREE}", gold))

    def test_style_gate_rejects_markdown_heading(self) -> None:
        gold = r"\boxed{2}"
        text = "### Step 1\n" + ("because addition. " * 40) + "\n\\boxed{2}"
        self.assertTrue(is_acceptable_solution(text, gold, min_chars=10))
        self.assertFalse(is_acceptable_solution(text, gold, min_chars=10, require_style=True))

    def test_hint_user_turn_never_stores_gold(self) -> None:
        gold = r"\boxed{2}"
        msgs = build_generation_messages("What is 1+1?", gold, True, allow_disagree=True)
        self.assertIn("DISAGREE", msgs[-1]["content"])
        src = {
            "messages": [{"role": "user", "content": "What is 1+1?"}],
            "question": "What is 1+1?",
            "solution": gold,
            "sample_id": "t3",
        }
        row = _make_sft_row(src, "Because 1+1=2.\n\\boxed{2}", hint_gold=True)
        self.assertFalse(user_turn_has_hint(row["messages"]))
        self.assertNotIn(HINT_MARK, row["messages"][0]["content"])


class SftQualityAuditTests(unittest.TestCase):
    def test_drops_grade_fail_and_hint_leak(self) -> None:
        from training.rl_data.audit_sft_solutions import audit_row

        ok = {
            "sample_id": "ok",
            "solution": r"\boxed{2}",
            "messages": [
                {"role": "user", "content": "1+1?"},
                {"role": "assistant", "content": "Reasoning here is long enough to pass the floor.\n\\boxed{2}"},
            ],
        }
        leak = {
            "sample_id": "leak",
            "solution": r"\boxed{2}",
            "messages": [
                {"role": "user", "content": f"1+1?\n{HINT_MARK}:\n\\boxed{2}"},
                {"role": "assistant", "content": "Reasoning here is long enough to pass the floor.\n\\boxed{2}"},
            ],
        }
        wrong = {
            "sample_id": "wrong",
            "solution": r"\boxed{2}",
            "messages": [
                {"role": "user", "content": "1+1?"},
                {"role": "assistant", "content": "Reasoning here is long enough to pass the floor.\n\\boxed{9}"},
            ],
        }
        self.assertFalse(audit_row(ok, min_chars=10)["drop"])
        leak_r = audit_row(leak, min_chars=10)
        self.assertTrue(leak_r["drop"])
        self.assertIn("hint_in_user", leak_r["flags"])
        wrong_r = audit_row(wrong, min_chars=10)
        self.assertTrue(wrong_r["drop"])
        self.assertIn("grade_fail", wrong_r["flags"])

    def test_heldout_is_dropped(self) -> None:
        from training.rl_data.audit_sft_solutions import audit_row

        row = {
            "sample_id": "h1",
            "solution": r"\boxed{1}",
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "long enough reasoning text for the check\n\\boxed{1}"},
            ],
        }
        r = audit_row(row, heldout_ids={"h1"}, min_chars=10)
        self.assertTrue(r["drop"])
        self.assertIn("heldout", r["flags"])


class SftRepairTests(unittest.TestCase):
    def test_strips_reference_answer_talk_and_still_grades(self) -> None:
        from training.rl_data.repair_sft_solutions import repair_row

        row = {
            "sample_id": "r1",
            "solution": r"\boxed{2}",
            "messages": [
                {"role": "user", "content": "1+1?"},
                {
                    "role": "assistant",
                    "content": (
                        "The reference answer suggests 2.\n\n"
                        "Because 1+1=2 we obtain the result.\n\\boxed{2}"
                    ),
                },
            ],
        }
        rec, status = repair_row(row)
        self.assertEqual(status, "repaired")
        asst = rec["messages"][-1]["content"]
        self.assertNotIn("reference answer", asst.lower())
        self.assertIn("\\boxed{2}", asst)


class HybridPassRateFilterTests(unittest.TestCase):
    def test_keeps_mixed_pass_rate_only(self) -> None:
        from training.swift.filter_swift_by_pass_rate import attach_and_filter

        prompts = [
            {"sample_id": "easy", "question": "q1", "solution": "1"},
            {"sample_id": "hard", "question": "q2", "solution": "2"},
            {"sample_id": "mid", "question": "q3", "solution": "3"},
        ]
        rollouts = (
            [{"sample_id": "easy", "acc": True}] * 8
            + [{"sample_id": "hard", "acc": False}] * 8
            + [{"sample_id": "mid", "acc": True}, {"sample_id": "mid", "acc": False}] * 4
        )
        kept, audit = attach_and_filter(prompts, rollouts, min_pass=0.05, max_pass=0.95)
        ids = {r["sample_id"] for r in kept}
        self.assertEqual(ids, {"mid"})
        self.assertEqual(audit["buckets"]["too_easy"], 1)
        self.assertEqual(audit["buckets"]["too_hard"], 1)


class HybridPilotAnalyzeTests(unittest.TestCase):
    def test_boxed_drop_is_fail(self) -> None:
        from training.swift.analyze_hybrid_pilot import evaluate_pilot

        metrics = [{"physics_format_rate": 0.4, "physics_answer_acc": 0.02, "physics_mixed_acc_group_rate": 0.1}] * 3
        metrics += [{"physics_format_rate": 0.1, "physics_answer_acc": 0.02, "physics_mixed_acc_group_rate": 0.1}] * 5
        report = evaluate_pilot(metrics=metrics, train=[{"completions/clipped_ratio": 0.2}], heldout={"answer_acc": 0.03})
        self.assertEqual(report["verdict"], "fail")
        self.assertIn("boxed_rate_dropped", report["flags"])


class SftManualEditTests(unittest.TestCase):
    def test_editorial_clean_strips_emoji_and_headers(self) -> None:
        from training.rl_data.build_sft_manual import editorial_clean

        raw = "✅ Derivation\n\nHere's the translated text:\n\n> 📌 note\n\nThe current is I.\n\\boxed{I}"
        cleaned = editorial_clean(raw)
        self.assertNotIn("✅", cleaned)
        self.assertNotIn("📌", cleaned)
        self.assertNotIn("Here's the translated text", cleaned)

    def test_user_fix_strips_83_420_answer_key(self) -> None:
        from training.rl_data.build_sft_manual import USER_FIX

        user = (
            "When in static equilibrium, the gas pressure equals $\\qquad$ (9).\n\n"
            "Figure 4\n\n"
            "(9) $P + \\frac{mg}{\\pi r^2}$"
        )
        fixed = USER_FIX["83_420"](user)
        self.assertNotIn(r"\frac{mg}{\pi r^2}", fixed)
        self.assertIn("equals $\\qquad$ (9).", fixed)
        self.assertNotIn("Figure 4", fixed)


class SpearmanTests(unittest.TestCase):
    def test_perfect_rank_correlation(self) -> None:
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]), 1.0)

    def test_inverse_rank_correlation(self) -> None:
        self.assertAlmostEqual(spearman([1.0, 2.0, 3.0], [30.0, 20.0, 10.0]), -1.0)


if __name__ == "__main__":
    unittest.main()
