from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_holdout_eval_samples import (
    DEFAULT_EXCLUDES,
    extract_holdout_samples,
    resolve_exclude_paths,
)


def _sample(index: int, question: str, *, correct: bool) -> dict:
    return {
        "index": index,
        "metadata": {"question": question},
        "response": "A detailed model response " * 8,
        "label": ["answer"],
        "reward": {"acc": correct, "score": 1.0 if correct else 0.0},
        "status": "completed",
    }


class ExtractHoldoutEvalSamplesTest(unittest.TestCase):
    def test_explicit_exclusions_are_additive_to_defaults(self) -> None:
        paths = resolve_exclude_paths([Path("data/development_set.json")])

        self.assertEqual(Path(DEFAULT_EXCLUDES[0]), paths[0])
        self.assertIn(Path("data/development_set.json"), paths)

    def test_excludes_questions_and_balances_classes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            exclude = root / "exclude.json"
            excluded_question = "Excluded physics question " * 4
            source.write_text(
                json.dumps(
                    [
                        {
                            "rollout_id": "r1",
                            "samples": [
                                _sample(1, excluded_question, correct=False),
                                _sample(2, "Wrong held out question " * 4, correct=False),
                                _sample(3, "Correct held out question " * 4, correct=True),
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            exclude.write_text(
                json.dumps([{"question": excluded_question}]),
                encoding="utf-8",
            )

            rows, report = extract_holdout_samples(
                input_path=source,
                exclude_paths=[exclude],
                wrong_size=1,
                correct_size=1,
                max_rollouts=0,
                seed=7,
            )

            self.assertEqual(2, len(rows))
            self.assertEqual(1, report["wrong_actual_size"])
            self.assertEqual(1, report["correct_actual_size"])
            self.assertTrue(report["question_keys_unique"])
            self.assertNotIn(excluded_question, {row["question"] for row in rows})


if __name__ == "__main__":
    unittest.main()
