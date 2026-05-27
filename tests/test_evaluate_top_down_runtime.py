from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from scripts.evaluate_top_down_runtime import evaluate_top_down_runtime


TMP_ROOT = Path("results/test_tmp")
TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"runtime_eval_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class EvaluateTopDownRuntimeTests(unittest.TestCase):
    def test_evaluate_top_down_runtime_summarizes_selection_quality(self) -> None:
        root = _case_dir()
        samples_path = root / "samples.json"
        catalog_path = root / "catalog.json"
        output_path = root / "runtime_report.json"
        samples_path.write_text(
            json.dumps(
                [
                    {"id": "s1", "question": "magnetic force", "prediction": "qvB", "answer": ""},
                    {"id": "s2", "question": "no match", "prediction": "unknown", "answer": ""},
                ]
            ),
            encoding="utf-8",
        )
        catalog_path.write_text(json.dumps({"metadata": {"catalog_type": "unified_rules_v2"}, "domains": []}), encoding="utf-8")

        class _FakeVerifier:
            def __init__(self) -> None:
                self.calls = 0

            def verify(self, sample):
                self.calls += 1
                if sample["id"] == "s1":
                    return {
                        "selection_strategy": "semantic_tree_selection",
                        "semantic_selection_error": "",
                        "retrieved_topics": [{"topic": "Magnetic Fields"}],
                        "retrieved_clusters": [{"cluster_id": "lorentz"}],
                        "retrieved_rules": [{"rule_id": "r1"}, {"rule_id": "r2"}],
                        "diagnostics": [{"type": "logic"}],
                    }
                return {
                    "selection_strategy": "semantic_tree_selection",
                    "semantic_selection_error": "",
                    "retrieved_topics": [{"topic": "Unknown"}],
                    "retrieved_clusters": [],
                    "retrieved_rules": [],
                    "diagnostics": [],
                }

        report = evaluate_top_down_runtime(
            samples_path=samples_path,
            catalog_path=catalog_path,
            output_path=output_path,
            limit=0,
            verifier_factory=lambda: _FakeVerifier(),
        )

        self.assertEqual(report["summary"]["sample_count"], 2)
        self.assertEqual(report["summary"]["semantic_tree_selection_count"], 2)
        self.assertEqual(report["summary"]["topic_selected_count"], 2)
        self.assertEqual(report["summary"]["cluster_selected_count"], 1)
        self.assertEqual(report["summary"]["rule_selected_count"], 1)
        self.assertEqual(report["summary"]["empty_rule_selection_count"], 1)
        self.assertEqual(report["summary"]["total_selected_rules"], 2)
        self.assertEqual(report["summary"]["diagnostic_sample_count"], 1)
        self.assertEqual(report["summary"]["empty_rule_sample_ids"], ["s2"])
        self.assertEqual(report["summary"]["high_rule_selection_sample_ids"], [])
        self.assertEqual(report["summary"]["broad_topic_selection_sample_ids"], [])
        self.assertEqual(report["summary"]["broad_cluster_selection_sample_ids"], [])
        self.assertTrue(output_path.exists())

    def test_evaluate_top_down_runtime_filters_by_sample_ids_before_limit(self) -> None:
        root = _case_dir()
        samples_path = root / "samples.json"
        catalog_path = root / "catalog.json"
        output_path = root / "runtime_report.json"
        samples_path.write_text(
            json.dumps(
                [
                    {"id": "s1", "question": "first", "prediction": "", "answer": ""},
                    {"id": "s2", "question": "second", "prediction": "", "answer": ""},
                    {"id": "s3", "question": "third", "prediction": "", "answer": ""},
                ]
            ),
            encoding="utf-8",
        )
        catalog_path.write_text(json.dumps({"metadata": {"catalog_type": "unified_rules_v2"}, "domains": []}), encoding="utf-8")

        class _FakeVerifier:
            def verify(self, sample):
                return {
                    "selection_strategy": "semantic_tree_selection",
                    "semantic_selection_error": "",
                    "retrieved_topics": [{"topic": sample["id"]}],
                    "retrieved_clusters": [],
                    "retrieved_rules": [{"rule_id": f"r_{sample['id']}"}],
                    "diagnostics": [],
                }

        report = evaluate_top_down_runtime(
            samples_path=samples_path,
            catalog_path=catalog_path,
            output_path=output_path,
            limit=1,
            sample_ids=["s2", "s3"],
            verifier_factory=lambda: _FakeVerifier(),
        )

        self.assertEqual(report["summary"]["sample_count"], 1)
        self.assertEqual(report["rows"][0]["sample_id"], "s2")
        self.assertEqual(report["rows"][0]["retrieved_rules"][0]["rule_id"], "r_s2")


if __name__ == "__main__":
    unittest.main()
