import json
import tempfile
import unittest
from pathlib import Path

from scripts.extract_incremental_test_samples import extract_incremental_samples


def _sample(index: int, question: str, *, acc: bool) -> dict:
    return {
        "index": index,
        "prompt": f"User: {question}\nAssistant:",
        "response": "A detailed but deliberately incorrect physical derivation. " * 3,
        "label": ["reference answer"],
        "reward": {"acc": acc, "score": 1.0 if acc else 0.0},
    }


class ExtractIncrementalTestSamplesTests(unittest.TestCase):
    def test_extracts_unique_wrong_questions_and_excludes_existing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "combined.json"
            excluded = root / "excluded.json"
            excluded_question = "Existing mechanics question " * 4
            new_question_a = "New electromagnetism question " * 4
            new_question_b = "New thermodynamics question " * 4
            source.write_text(
                json.dumps(
                    [
                        {
                            "rollout_id": 1,
                            "samples": [
                                _sample(0, excluded_question, acc=False),
                                _sample(1, new_question_a, acc=False),
                                _sample(2, new_question_a, acc=False),
                                _sample(3, "Correct answer question " * 4, acc=True),
                            ],
                        },
                        {
                            "rollout_id": 2,
                            "samples": [_sample(0, new_question_b, acc=False)],
                        },
                    ]
                ),
                encoding="utf-8",
            )
            excluded.write_text(
                json.dumps([{"question": excluded_question}]),
                encoding="utf-8",
            )

            rows, report = extract_incremental_samples(
                input_path=source,
                exclude_paths=[excluded],
                target_size=2,
                max_rollouts=0,
                seed=7,
            )

            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["meta"]["question_key"] for row in rows}), 2)
            self.assertTrue(all(row["source_reward_acc"] is False for row in rows))
            self.assertNotIn(excluded_question, {row["question"] for row in rows})
            self.assertEqual(report["eligible_unique_questions"], 2)
            self.assertTrue(report["all_selected_are_wrong"])


if __name__ == "__main__":
    unittest.main()
