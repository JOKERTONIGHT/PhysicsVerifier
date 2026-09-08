from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.rl_data.build_rft_from_rollouts import select_rollouts
from training.rl_data.drop_disagree_prompts import disagree_ids, filter_pool
from training.rl_data.audit_heldout_gold import heldout_to_prompt


class RftSelectTests(unittest.TestCase):
    def test_picks_length_near_target(self) -> None:
        rows = []
        for i, text in enumerate(["short \\boxed{1}", ("x" * 4500) + "\\boxed{1}", ("y" * 9000) + "\\boxed{1}"]):
            rows.append(
                {
                    "sample_id": "a",
                    "acc": True,
                    "response": "Energy is conserved.\n" + text,
                    "rollout_index": i,
                    "question": "Find v.",
                    "solution": r"\boxed{1}",
                }
            )
        kept, audit = select_rollouts(rows, max_per_id=1, target_len=4600)
        self.assertEqual(audit["n_ids_with_correct"], 1)
        self.assertEqual(len(kept), 1)
        self.assertIn("x" * 20, kept[0]["response"])

    def test_skips_incorrect(self) -> None:
        rows = [{"sample_id": "a", "acc": False, "part_frac": 0.0, "response": "\\boxed{9}"}]
        kept, audit = select_rollouts(rows)
        self.assertEqual(kept, [])
        self.assertEqual(audit["n_correct"], 0)

    def test_keeps_qwen_markdown_in_rft_mode(self) -> None:
        text = "### Setup\n" + ("Energy is conserved. " * 40) + "\n\\boxed{1}"
        rows = [{"sample_id": "a", "acc": True, "response": text, "rollout_index": 0}]
        kept, audit = select_rollouts(rows, max_per_id=1)
        self.assertEqual(audit["n_correct"], 1)
        self.assertEqual(audit["n_style_drop"], 0)
        self.assertEqual(len(kept), 1)


class SeedLongchainTests(unittest.TestCase):
    def test_seeds_correct_rollout_for_prompt_id(self) -> None:
        from training.rl_data.seed_longchain_from_rollouts import seed

        prompts = [{"sample_id": "a", "question": "Find v.", "solution": r"\boxed{1}"}]
        rollouts = [
            {
                "sample_id": "a",
                "acc": True,
                "response": ("Energy is conserved. " * 200) + "\n\\boxed{1}",
                "question": "Find v.",
            },
            {
                "sample_id": "b",
                "acc": True,
                "response": ("Energy is conserved. " * 200) + "\n\\boxed{2}",
            },
        ]
        rows, audit = seed(prompts, rollouts, done=set(), target_len=4500)
        self.assertEqual(audit["n_seeded"], 1)
        self.assertEqual(rows[0]["sample_id"], "a")
        self.assertEqual(rows[0]["generator"], "base_rollout_anchor")

    def test_skips_done_ids(self) -> None:
        from training.rl_data.seed_longchain_from_rollouts import seed

        prompts = [{"sample_id": "a", "question": "q", "solution": r"\boxed{1}"}]
        rollouts = [
            {
                "sample_id": "a",
                "acc": True,
                "response": ("Energy is conserved. " * 200) + "\n\\boxed{1}",
            }
        ]
        rows, _ = seed(prompts, rollouts, done={"a"})
        self.assertEqual(rows, [])


class DropDisagreeTests(unittest.TestCase):
    def test_filters_matching_ids(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "d.jsonl"
            path.write_text(json.dumps({"sample_id": "bad"}) + "\n", encoding="utf-8")
            drop = disagree_ids(path)
            kept, n = filter_pool(
                [{"sample_id": "bad"}, {"sample_id": "ok"}],
                drop,
            )
            self.assertEqual(n, 1)
            self.assertEqual(kept[0]["sample_id"], "ok")


class HeldoutConvertTests(unittest.TestCase):
    def test_joins_list_labels(self) -> None:
        row = {
            "input": [{"role": "system", "content": "sys"}, {"role": "user", "content": "Find a and b."}],
            "label": [r"\boxed{1}", r"\boxed{2}"],
            "metadata": {"sample_id": "h1", "source": "heldout"},
        }
        out = heldout_to_prompt(row)
        self.assertEqual(out["sample_id"], "h1")
        self.assertIn("1", out["solution"])
        self.assertIn("2", out["solution"])
        self.assertEqual(out["question"], "Find a and b.")


if __name__ == "__main__":
    unittest.main()
