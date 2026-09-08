from __future__ import annotations

import unittest

from training.rl_data.screen_training_data import (
    gold_fit_drop_reason,
    looks_multi_blank,
    prompt_drop_reason,
    sft_row_drop_reason,
    sft_style_drop_reason,
    visual_drop_reason,
)


class VisualScreenTests(unittest.TestCase):
    def test_drops_explicit_figure_reference(self) -> None:
        self.assertEqual(
            visual_drop_reason("As shown in the figure, a railgun consists of two rails."),
            "visual_input",
        )
        self.assertEqual(visual_drop_reason("See Figure 7. The I-V curve is given."), "visual_input")
        self.assertEqual(
            visual_drop_reason("The maximum torque (see the following figure, a larger copy is on an extra sheet)."),
            "visual_input",
        )
        self.assertEqual(
            visual_drop_reason("A wall, as shown in the right figure, has a fixed temperature difference."),
            "visual_input",
        )
        self.assertEqual(
            visual_drop_reason("As shown in the gas pressure-volume diagram, a diatomic gas undergoes a cycle."),
            "visual_input",
        )
        self.assertEqual(visual_drop_reason("如图所示，电路中电阻为 R。"), "visual_input")

    def test_keeps_significant_figures_and_text_only_stems(self) -> None:
        self.assertIsNone(visual_drop_reason("Give the answer to three significant figures."))
        self.assertIsNone(
            visual_drop_reason(
                "A satellite of mass m orbits Earth. Find the orbital speed in terms of G, M, and r."
            )
        )

    def test_prompt_drop_known_unusable_and_concat(self) -> None:
        self.assertEqual(prompt_drop_reason({"sample_id": "83_148", "question": "Find i(t)."}), "known_unusable")
        concat = {
            "sample_id": "ok",
            "question": (
                "Question: A satellite orbits at 160 km.\n\n"
                + ("x" * 80)
                + "\nQuestion: Express T1 and T2 in terms of x1."
            ),
            "solution": r"\boxed{1}",
        }
        self.assertEqual(prompt_drop_reason(concat), "concatenated_stem")

    def test_sft_drops_gold_fit_talk(self) -> None:
        row = {
            "sample_id": "x",
            "question": "A block of mass m slides on ice. Find v.",
            "solution": r"\boxed{\sqrt{L}}",
            "messages": [
                {"role": "user", "content": "A block of mass m slides on ice. Find v."},
                {
                    "role": "assistant",
                    "content": "We are not given T0 but the known result 1310 m implies we take 288 K.\n\\boxed{\\sqrt{L}}",
                },
            ],
        }
        self.assertEqual(sft_row_drop_reason(row), "gold_fit")
        self.assertIsNotNone(gold_fit_drop_reason(row["messages"][-1]["content"]))
        self.assertIsNone(gold_fit_drop_reason("Energy is conserved, so v=sqrt(2gh).\nThis matches the reference.\n\\boxed{\\sqrt{2gh}}"))
        numbered = {
            "sample_id": "proc",
            "question": "The cycle has two steps.\n1. A mass is placed on the piston.\n2. The gas expands isothermally.\nFind the work.",
            "solution": r"\boxed{W}",
        }
        self.assertIsNone(prompt_drop_reason(numbered))


class StyleGateTests(unittest.TestCase):
    def test_multi_box_and_heading(self) -> None:
        body = "Energy is conserved.\n" + ("x" * 400)
        self.assertEqual(sft_style_drop_reason(body + "\n\\boxed{1}\n\\boxed{2}"), "multi_box")
        self.assertEqual(
            sft_style_drop_reason("### Step 1\n" + body + "\n\\boxed{1}"),
            "md_heading",
        )

    def test_truncated_and_ramble(self) -> None:
        self.assertEqual(sft_style_drop_reason("partial derivation\\boxed{1}\n### "), "truncated")
        ramble = ("Energy is conserved. " * 40) + "\\boxed{1}\n" + ("after " * 120)
        self.assertEqual(sft_style_drop_reason(ramble), "ramble_after_box")

    def test_multi_blank_stem_drops_sft_row(self) -> None:
        q = r"Find v = $\qquad$ (6) m/s and R = $\qquad$ (7) m."
        self.assertTrue(looks_multi_blank(q))
        row = {
            "sample_id": "blank2",
            "question": q,
            "solution": r"\boxed{198}",
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": ("Derivation. " * 40) + "\n\\boxed{198}"},
            ],
        }
        self.assertEqual(sft_row_drop_reason(row), "multi_blank_stem")
        self.assertIsNone(prompt_drop_reason(row))

    def test_rft_mode_allows_multi_box_and_long(self) -> None:
        body = "Energy is conserved.\n" + ("step. " * 80)
        text = body + "\n\\boxed{1}\nthen\n\\boxed{2}"
        self.assertEqual(sft_style_drop_reason(text), "multi_box")
        self.assertIsNone(sft_style_drop_reason(text, mode="rft"))
        long_ok = ("Derive. " * 900) + "\n\\boxed{1}"
        self.assertEqual(sft_style_drop_reason(long_ok), "too_long")
        self.assertIsNone(sft_style_drop_reason(long_ok, mode="rft"))
        loop = "the uncertainty in the measurement. " * 40
        self.assertEqual(sft_style_drop_reason("\\boxed{1}\n" + (loop * 3), mode="rft"), "repetition")
        heading = "### Final\n" + ("Energy is conserved. " * 40) + "\n\\boxed{1}"
        self.assertEqual(sft_style_drop_reason(heading), "md_heading")
        self.assertIsNone(sft_style_drop_reason(heading, mode="rft"))
        emoji = ("Energy is conserved. " * 40) + "✅\n\\boxed{1}"
        self.assertEqual(sft_style_drop_reason(emoji), "emoji")
        self.assertIsNone(sft_style_drop_reason(emoji, mode="rft"))


if __name__ == "__main__":
    unittest.main()
