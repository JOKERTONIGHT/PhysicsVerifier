from __future__ import annotations

import unittest

from scripts.audit_verifier_context_sources import _surface_mismatches, _topic_text_surfaces


class VerifierContextSourceAuditTests(unittest.TestCase):
    @staticmethod
    def _payload(**fields):
        return {"evidence": fields}

    def test_detects_precondition_satisfied_only_by_solution(self) -> None:
        mixed = self._payload(precondition_hits=["constant velocity"])
        problem = self._payload(precondition_hits=[])
        solution = self._payload(precondition_hits=["constant velocity"])
        self.assertEqual(
            _surface_mismatches(mixed, problem, solution),
            {"precondition_from_solution_only": ["constant velocity"]},
        )

    def test_detects_violation_signature_coming_only_from_problem(self) -> None:
        mixed = self._payload(violation_signature_hits=["use latent heat at 100 C"])
        problem = self._payload(violation_signature_hits=["use latent heat at 100 C"])
        solution = self._payload(violation_signature_hits=[])
        self.assertEqual(
            _surface_mismatches(mixed, problem, solution),
            {"violation_from_problem_only": ["use latent heat at 100 C"]},
        )

    def test_does_not_flag_signal_present_on_expected_surface(self) -> None:
        mixed = self._payload(
            precondition_hits=["nearly circular orbit"],
            violation_signature_hits=["uses constant drag"],
        )
        problem = self._payload(precondition_hits=["nearly circular orbit"])
        solution = self._payload(violation_signature_hits=["uses constant drag"])
        self.assertEqual(_surface_mismatches(mixed, problem, solution), {})

    def test_topic_surfaces_keep_problem_and_solution_sources_separate(self) -> None:
        surfaces = _topic_text_surfaces(
            {
                "question": "QUESTION_ONLY_TOKEN",
                "context": "CONTEXT_ONLY_TOKEN",
                "prediction": "SOLUTION_PREFIX_AND_SUFFIX",
            },
            {"topic_prediction_max_chars": 15},
        )
        self.assertIn("QUESTION_ONLY_TOKEN", surfaces["current"])
        self.assertIn("SOLUTION_PREFIX", surfaces["current"])
        self.assertNotIn("SOLUTION_PREFIX", surfaces["problem_only"])
        self.assertNotIn("QUESTION_ONLY_TOKEN", surfaces["solution_only"])
        self.assertIn("SOLUTION_PREFIX", surfaces["solution_only"])
        self.assertNotIn("AND_SUFFIX", surfaces["solution_only"])


if __name__ == "__main__":
    unittest.main()
