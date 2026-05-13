import unittest
from pathlib import Path

from scripts.combined_language_io import iter_rollout_batches


class CombinedLanguageIoTests(unittest.TestCase):
    def test_iter_mini_fixture_yields_two_rollouts(self) -> None:
        p = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "combined_rollouts_mini.json"
        batches = list(iter_rollout_batches(p))
        self.assertEqual(len(batches), 2)
        self.assertEqual(batches[0].get("rollout_id"), 99)
        self.assertEqual(len(batches[0].get("samples") or []), 1)


if __name__ == "__main__":
    unittest.main()
