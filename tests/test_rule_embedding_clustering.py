from __future__ import annotations

import unittest

from scripts.run_rule_embedding_clustering import _cluster_topic_rules


class RuleEmbeddingClusteringTests(unittest.TestCase):
    def test_clusters_rules_within_topic_by_cosine_threshold(self) -> None:
        rules = [
            {"rule_id": "a", "title": "A", "summary": "one", "trigger": "x"},
            {"rule_id": "b", "title": "B", "summary": "one", "trigger": "x"},
            {"rule_id": "c", "title": "C", "summary": "two", "trigger": "y"},
        ]
        embeddings = {
            "a": [1.0, 0.0],
            "b": [0.99, 0.01],
            "c": [0.0, 1.0],
        }

        clusters, residual = _cluster_topic_rules(
            rules,
            embeddings,
            threshold=0.95,
            min_cluster_size=2,
        )

        self.assertEqual(len(clusters), 1)
        self.assertEqual(clusters[0]["rule_ids"], ["a", "b"])
        self.assertEqual(residual, ["c"])


if __name__ == "__main__":
    unittest.main()
