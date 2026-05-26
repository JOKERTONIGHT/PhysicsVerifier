from __future__ import annotations

import json
import unittest
import uuid
from pathlib import Path

from scripts.analyze_rule_embedding_clusters import analyze_embedding_clusters


TMP_ROOT = Path("results/test_tmp")
TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"embedding_cluster_analysis_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class AnalyzeRuleEmbeddingClustersTests(unittest.TestCase):
    def test_analyze_embedding_clusters_reports_coverage(self) -> None:
        root = _case_dir()
        input_path = root / "clusters.json"
        output_path = root / "report.json"
        input_path.write_text(
            json.dumps(
                {
                    "metadata": {"generator": "topic_local_rule_embedding_clustering_v1"},
                    "topics": [
                        {
                            "topic_key": "mechanics::kinematics",
                            "rule_count": 10,
                            "clusters": [
                                {"cluster_id": "c1", "rule_ids": ["r1", "r2", "r3", "r4"]},
                                {"cluster_id": "c2", "rule_ids": ["r5", "r6"]},
                            ],
                            "residual_rule_ids": ["r7", "r8", "r9", "r10"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        report = analyze_embedding_clusters(input_path=input_path, output_path=output_path)

        self.assertTrue(report["ready_for_labeling"])
        self.assertEqual(report["topic_count"], 1)
        self.assertEqual(report["total_clustered_rule_count"], 6)
        self.assertEqual(report["clustered_rule_ratio"], 0.6)
        self.assertTrue(output_path.exists())

    def test_strict_fails_when_clustered_ratio_too_low(self) -> None:
        root = _case_dir()
        input_path = root / "clusters.json"
        input_path.write_text(
            json.dumps(
                {
                    "topics": [
                        {
                            "topic_key": "mechanics::dynamics",
                            "rule_count": 10,
                            "clusters": [{"cluster_id": "c1", "rule_ids": ["r1"]}],
                            "residual_rule_ids": ["r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10"],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaises(SystemExit):
            analyze_embedding_clusters(
                input_path=input_path,
                min_clustered_rule_ratio=0.3,
                strict=True,
            )


if __name__ == "__main__":
    unittest.main()
