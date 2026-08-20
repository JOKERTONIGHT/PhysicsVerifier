from __future__ import annotations

import unittest

from scripts.audit_correct_eval_candidates import _extract_json_object, _normalize_audit


class AuditCorrectEvalCandidatesTest(unittest.TestCase):
    def test_extracts_fenced_or_prefixed_json(self) -> None:
        payload = _extract_json_object('Result: {"verdict":"correct","errors":[],"rationale":"sound"}')
        self.assertEqual("correct", payload["verdict"])

    def test_correct_with_errors_becomes_uncertain(self) -> None:
        audit = _normalize_audit(
            {"verdict": "correct", "errors": ["bad sign"], "rationale": "inconsistent response"}
        )
        self.assertEqual("uncertain", audit["verdict"])


if __name__ == "__main__":
    unittest.main()
