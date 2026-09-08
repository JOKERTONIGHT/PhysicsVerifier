from __future__ import annotations

import unittest

from training.rl_data.clean_longchain_anchors import (
    clean_assistant,
    clean_rows,
    keep_last_boxed,
)


def _anchor(text: str, gold: str = r"\boxed{2}", sid: str = "a") -> dict:
    return {
        "sample_id": sid,
        "generator": "base_rollout_anchor",
        "solution": gold,
        "question": "Find v.",
        "messages": [
            {"role": "user", "content": "Find v."},
            {"role": "assistant", "content": text},
        ],
    }


class KeepLastBoxedTests(unittest.TestCase):
    def test_unwraps_earlier_boxes(self) -> None:
        text = r"first \boxed{1} then \boxed{2}"
        out = keep_last_boxed(text)
        self.assertEqual(out.count(r"\boxed"), 1)
        self.assertIn("1", out)
        self.assertIn(r"\boxed{2}", out)


class CleanAssistantTests(unittest.TestCase):
    def test_strips_heading_emoji_and_extra_box(self) -> None:
        body = " ".join(f"Step {i}: apply conservation of energy to the block." for i in range(80))
        text = "### Step 1\n" + body + "\n✅\n\\boxed{1}\nthen\n\\boxed{2}\n"
        cleaned = clean_assistant(text)
        self.assertNotIn("###", cleaned)
        self.assertNotIn("✅", cleaned)
        self.assertEqual(cleaned.count("\\boxed"), 1)
        self.assertIn("\\boxed{2}", cleaned)


class CleanRowsTests(unittest.TestCase):
    def test_keeps_cleaned_anchor_and_api(self) -> None:
        body = " ".join(f"Step {i}: apply conservation of energy to the block." for i in range(80))
        text = "### Setup\n" + body + "\n✅\n\\boxed{1}\nend\n\\boxed{2}\n"
        api = {
            "sample_id": "api1",
            "generator": "api",
            "solution": r"\boxed{9}",
            "messages": [{"role": "assistant", "content": "short"}],
        }
        kept, report = clean_rows([_anchor(text), api], min_chars=400)
        self.assertEqual(report["n_cleaned"], 1)
        self.assertEqual(report["n_passthrough"], 1)
        self.assertEqual(len(kept), 2)
        asst = kept[0]["messages"][-1]["content"]
        self.assertEqual(asst.count("\\boxed"), 1)
        self.assertEqual(kept[0]["generator"], "base_rollout_anchor_clean")
        self.assertEqual(kept[1]["sample_id"], "api1")

    def test_drops_gold_mismatch(self) -> None:
        body = " ".join(f"Step {i}: apply conservation of energy to the block." for i in range(80))
        text = body + "\n\\boxed{2}\n"
        kept, report = clean_rows([_anchor(text, gold=r"\boxed{9}")], min_chars=400)
        self.assertEqual(kept, [])
        self.assertEqual(report["dropped_reasons"].get("gold_mismatch"), 1)

    def test_drops_repetitive_tail(self) -> None:
        pad = " ".join(f"Step {i}: apply conservation of energy to the block." for i in range(40))
        loop = "LOOPMARK the same sentence again and again. "
        text = pad + "\n" + (loop * 30) + "\n\\boxed{2}\n"
        kept, report = clean_rows([_anchor(text)], min_chars=400)
        if report["n_dropped"]:
            self.assertGreaterEqual(report["dropped_reasons"].get("repetition", 0), 1)
            return
        self.assertEqual(len(kept), 1)
        body = kept[0]["messages"][-1]["content"]
        self.assertLess(body.count(loop), 6)


if __name__ == "__main__":
    unittest.main()
