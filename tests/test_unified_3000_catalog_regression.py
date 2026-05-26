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

    def test_seeded_170364_heat_transfer_rules_are_present(self) -> None:
        heat_topic = None
        for domain in self.catalog["domains"]:
            if domain["name"] != "Thermodynamics & Statistical Physics":
                continue
            for topic in domain["topics"]:
                if topic["name"] == "Heat Transfer (Conduction, Convection, Radiation)":
                    heat_topic = topic
                    break
        self.assertIsNotNone(heat_topic)

        rules = heat_topic["rules"]
        titles = {rule["title"] for rule in rules}
        self.assertIn("变功率冷却时间积分规则", titles)
        self.assertIn("图表信息提取完整性校验", titles)

        cooling_cluster = next(
            cluster
            for cluster in heat_topic["scenario_clusters"]
            if cluster["id"] == "heating_cooling_and_capacity_model"
        )
        cooling_rule_titles = {
            rule["title"]
            for rule in rules
            if rule["rule_id"] in set(cooling_cluster["rule_ids"])
        }
        self.assertIn("变功率冷却时间积分规则", cooling_rule_titles)

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
