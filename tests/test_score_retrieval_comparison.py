import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.score_retrieval_comparison import prepare, score, sha, write_new


class PhysicalMetricsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / "inputs.json"
        self.refs = self.root / "refs.json"
        self.run = self.root / "checker"
        write_new(self.inputs, [{"id": "q", "question": "Q", "prediction": "A"}])
        write_new(self.refs, [{"id": "q", "errors": [{"source_gt": {"error_id": "old", "error_text": "Unreviewed"}}]}])
        write_new(self.run / "manifest.json", {"input_sha256": sha(self.inputs), "configuration_sha256": "cfg", "model": "qwen"})
        self.result_path = self.run / "arm2" / hashlib.sha256(b"q").hexdigest()[:16] / "result.json"
        write_new(self.result_path, {"id": "q", "arm": 2, "model": "qwen", "configuration_sha256": "cfg", "status": "completed", "trace": {"diagnostics": [{"message": "one"}, {"message": "two"}]}})
        self.expected = prepare(self.inputs, self.refs, self.run, [2])
        self.review = copy.deepcopy(self.expected)
        self.review["reviewer"] = "test reviewer"
        case = self.review["cases"][0]
        case.update(reference_review_complete=True, reference_review_notes="Independently checked", adjudicated_errors=[{"error_id": "a", "error_text": "First error"}, {"error_id": "b", "error_text": "Second error"}])
        for d in case["arms"]["2"]["diagnostics"]:
            d.update(review_complete=True, rationale="Physical reason checked", matched_error_ids=["a"])

    def metrics(self):
        return score(self.review, self.expected)["arms"]["2"]

    def test_duplicate_diagnoses_are_false_positive(self):
        m = self.metrics()
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 1, 1))
        self.assertEqual((m["precision"], m["recall"], m["f1"]), (.5, .5, .5))

    def test_one_to_one_assignment_does_not_depend_on_greedy_order(self):
        self.review["cases"][0]["arms"]["2"]["diagnostics"][0]["matched_error_ids"] = ["a", "b"]
        self.assertEqual(self.metrics()["tp"], 2)

    def test_wrong_answer_does_not_make_unrelated_diagnosis_true_positive(self):
        for d in self.review["cases"][0]["arms"]["2"]["diagnostics"]:
            d["matched_error_ids"] = []
        self.assertEqual((self.metrics()["tp"], self.metrics()["fp"], self.metrics()["fn"]), (0, 2, 2))

    def test_missing_failed_or_unreviewed_cases_block_metrics(self):
        for mutation in ("failed", "missing", "pending", "model", "removed"):
            with self.subTest(mutation=mutation):
                reviewed, expected = copy.deepcopy(self.review), copy.deepcopy(self.expected)
                if mutation in ("failed", "missing"):
                    expected["cases"][0]["arms"]["2"]["status"] = mutation
                elif mutation == "pending":
                    reviewed["cases"][0]["reference_review_complete"] = False
                elif mutation == "model":
                    reviewed["reviewer_kind"] = "model_assisted"
                else:
                    reviewed["cases"] = []
                with self.assertRaises(ValueError):
                    score(reviewed, expected)

    def test_changed_diagnosis_or_unknown_error_is_rejected(self):
        for field, value in (("diagnostic", {"message": "changed"}), ("matched_error_ids", ["unknown"])):
            reviewed = copy.deepcopy(self.review)
            reviewed["cases"][0]["arms"]["2"]["diagnostics"][0][field] = value
            with self.assertRaises(ValueError):
                score(reviewed, self.expected)

    def test_completed_empty_is_valid_and_missing_is_not_empty(self):
        result = json.loads(self.result_path.read_text())
        result["trace"]["diagnostics"] = []
        self.result_path.write_text(json.dumps(result))
        expected = prepare(self.inputs, self.refs, self.run, [2])
        reviewed = copy.deepcopy(self.review)
        reviewed["cases"][0]["arms"] = copy.deepcopy(expected["cases"][0]["arms"])
        m = score(reviewed, expected)["arms"]["2"]
        self.assertIsNone(m["precision"])
        self.assertEqual((m["recall"], m["fn"]), (0, 2))
        self.result_path.unlink()
        self.assertEqual(prepare(self.inputs, self.refs, self.run, [2])["cases"][0]["arms"]["2"]["status"], "missing")

    def test_changed_input_and_result_model_are_rejected(self):
        result = json.loads(self.result_path.read_text())
        result["model"] = "other"
        self.result_path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "model"):
            prepare(self.inputs, self.refs, self.run, [2])
        self.inputs.write_text(self.inputs.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "input hash"):
            prepare(self.inputs, self.refs, self.run, [2])

    def test_review_is_never_overwritten(self):
        with self.assertRaises(FileExistsError):
            write_new(self.refs, [])


if __name__ == "__main__":
    unittest.main()
