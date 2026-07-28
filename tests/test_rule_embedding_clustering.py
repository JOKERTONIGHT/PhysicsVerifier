from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.run_rule_embedding_clustering import (
    _cache_payload,
    _cluster_topic_rules,
    _embed_rules,
    _load_valid_cached_embeddings,
)


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

    def test_missing_embedding_remains_reachable_as_residual(self) -> None:
        rules = [{"rule_id": "a"}, {"rule_id": "missing"}]

        clusters, residual = _cluster_topic_rules(
            rules,
            {"a": [1.0, 0.0]},
            threshold=0.95,
            min_cluster_size=2,
        )

        self.assertEqual(clusters, [])
        self.assertCountEqual(residual, ["a", "missing"])

    def test_cache_is_reused_only_for_same_model_and_text(self) -> None:
        rules = [
            {
                "rule_id": "r1",
                "topic_key": "mechanics::kinematics",
                "embedding_text": "original text",
            }
        ]
        cache = _cache_payload(model="embedding-v1", rules=rules, embeddings={"r1": [1.0, 0.0]})

        self.assertEqual(
            _load_valid_cached_embeddings(cache=cache, model="embedding-v1", rules=rules),
            {"r1": [1.0, 0.0]},
        )
        changed_rules = [dict(rules[0], embedding_text="changed text")]
        self.assertEqual(
            _load_valid_cached_embeddings(
                cache=cache,
                model="embedding-v1",
                rules=changed_rules,
            ),
            {},
        )
        self.assertEqual(
            _load_valid_cached_embeddings(cache=cache, model="embedding-v2", rules=rules),
            {},
        )

    def test_embedding_response_must_cover_every_input(self) -> None:
        class FakeEmbeddings:
            @staticmethod
            def create(**_):
                return SimpleNamespace(
                    data=[SimpleNamespace(index=0, embedding=[1.0, 0.0])]
                )

        client = SimpleNamespace(embeddings=FakeEmbeddings())
        rules = [
            {"rule_id": "r1", "embedding_text": "one"},
            {"rule_id": "r2", "embedding_text": "two"},
        ]

        with self.assertRaisesRegex(RuntimeError, "1 vectors for 2 inputs"):
            _embed_rules(
                client=client,
                model="embedding-v1",
                rules=rules,
                batch_size=2,
                existing={},
            )


if __name__ == "__main__":
    unittest.main()
