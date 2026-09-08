from __future__ import annotations

import unittest

from training.rl_data.answer_equiv import answers_equivalent


def _ok(cand: str, gold: str) -> tuple[bool, str]:
    return answers_equivalent(cand, gold)


class AnswerEquivTests(unittest.TestCase):
    def test_mathrm_tilde_newton(self) -> None:
        gold = r"$\boxed{2.41 \times 10^{-4} \mathrm{~N}}$"
        self.assertTrue(_ok(r"\boxed{2.41\times10^{-4}\ \text{N}}", gold)[0])
        self.assertTrue(_ok(r"\boxed{2.41\times10^{-4}}", gold)[0])

    def test_scientific_vs_plain_with_unit(self) -> None:
        self.assertTrue(_ok(r"\boxed{1.2631\times10^{4}\ \mathrm{N}}", "12631 N")[0])
        self.assertTrue(_ok(r"\boxed{12631}", "12631 N")[0])

    def test_si_prefix_length(self) -> None:
        self.assertTrue(_ok(r"\boxed{2.6\times10^{-5}\ \text{m}}", "26 μm")[0])
        self.assertTrue(_ok(r"\boxed{26}", "26 μm")[0])

    def test_bare_number_vs_ohm(self) -> None:
        self.assertTrue(_ok(r"\boxed{2.5\ \Omega}", "2.5")[0])
        self.assertTrue(_ok(r"\boxed{2.5}", "2.5")[0])

    def test_strips_lhs_equals(self) -> None:
        self.assertTrue(_ok(r"\boxed{1.65a}", r"$r=1.65a$")[0])
        self.assertTrue(_ok(r"\boxed{0.8U}", r"$U_{\min } \approx 0.8 U$")[0])

    def test_mev_unit(self) -> None:
        self.assertTrue(_ok(r"\boxed{1.02}", r"1.02 \mathrm{MeV}")[0])
        self.assertTrue(_ok(r"\boxed{1.02\,\mathrm{MeV}}", r"1.02 \mathrm{MeV}")[0])

    def test_rel_tol_rejects_one_percent(self) -> None:
        self.assertFalse(_ok(r"\boxed{200}", "198")[0])
        self.assertFalse(_ok(r"\boxed{99}", "198")[0])

    def test_percent_is_not_auto_converted(self) -> None:
        self.assertFalse(_ok(r"\boxed{4.34\%}", "0.0434")[0])

    def test_verl_still_handles_fractions(self) -> None:
        self.assertTrue(_ok(r"\boxed{\frac{1}{2}}", "0.5")[0])
        self.assertEqual(_ok(r"\boxed{\frac{1}{2}}", "0.5")[1], "verl")

    def test_degree_symbol(self) -> None:
        self.assertTrue(_ok(r"\boxed{67.8^\circ}", r"$\boxed{67.8^{\circ}}$")[0])
        self.assertTrue(_ok(r"\boxed{67.8}", r"$\boxed{67.8^{\circ}}$")[0])


if __name__ == "__main__":
    unittest.main()
