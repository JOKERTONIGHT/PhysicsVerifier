from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class Unified3000CatalogRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = json.loads((REPO_ROOT / "catalogs" / "rules_unified_3000.json").read_text(encoding="utf-8"))

    def test_cluster_rule_ids_reference_rules_in_the_same_topic(self) -> None:
        invalid_references = []
        for domain in self.catalog["domains"]:
            for topic in domain["topics"]:
                topic_rule_ids = {rule["rule_id"] for rule in topic.get("rules", [])}
                for cluster in topic.get("scenario_clusters", []):
                    unknown_ids = set(cluster.get("rule_ids", [])) - topic_rule_ids
                    if unknown_ids:
                        invalid_references.append(
                            f"{domain['name']}::{topic['name']}::{cluster['id']}::{sorted(unknown_ids)}"
                        )

        self.assertEqual(invalid_references, [])

    def test_no_topic_has_duplicate_cluster_ids(self) -> None:
        duplicate_topics = []
        for domain in self.catalog["domains"]:
            for topic in domain["topics"]:
                ids = [cluster["id"] for cluster in topic.get("scenario_clusters", [])]
                duplicates = [cluster_id for cluster_id, count in Counter(ids).items() if count > 1]
                if duplicates:
                    duplicate_topics.append(f"{domain['name']}::{topic['name']}::{','.join(duplicates)}")

        self.assertEqual(duplicate_topics, [])


if __name__ == "__main__":
    unittest.main()
