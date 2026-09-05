from __future__ import annotations

import unittest

from training.rl_data.audit_sft_solutions import audit_row
from training.rl_data.generate_sft_solutions import HINT_MARK


class AuditSftTests(unittest.TestCase):
    def test_drops_hint_in_user(self) -> None:
        row = {
            "sample_id": "1",
            "solution": "2",
            "messages": [
                {"role": "user", "content": f"Find T.\n{HINT_MARK}: 2"},
                {"role": "assistant", "content": r"work\n\boxed{2}"},
            ],
        }
        detail = audit_row(row, min_chars=0)
        self.assertIn("hint_in_user", detail["flags"])
        self.assertTrue(detail["drop"])

    def test_warns_too_short_but_keeps(self) -> None:
        row = {
            "sample_id": "2",
            "solution": "",
            "question": "A mass m on a spring of k. Find T.",
            "messages": [
                {"role": "user", "content": "A mass m on a spring of k. Find T."},
                {"role": "assistant", "content": r"T=2\pi\sqrt{m/k}\n\boxed{2\pi\sqrt{m/k}}"},
            ],
        }
        detail = audit_row(row, min_chars=400)
        self.assertIn("too_short", detail["flags"])
        self.assertFalse(detail["drop"])


if __name__ == "__main__":
    unittest.main()
