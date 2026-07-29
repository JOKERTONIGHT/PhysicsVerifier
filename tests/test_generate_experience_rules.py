import unittest

from scripts.generate_experience_rules import (
    TopicItem,
    _canonical_topic,
    _normalize_semantic_result,
)


class GenerateExperienceRulesTests(unittest.TestCase):
    def test_exact_topic_is_preserved(self) -> None:
        topics = [
            TopicItem("Mechanics", "Rotational Motion"),
            TopicItem("Electromagnetism", "Electric Fields and Potential"),
        ]
        selected = _canonical_topic(
            {"domain": "Mechanics", "topic": "Rotational Motion"},
            topics,
        )
        self.assertEqual(selected, topics[0])

    def test_out_of_catalog_label_maps_to_existing_topic(self) -> None:
        topics = [
            TopicItem("Mechanics", "Rotational Motion"),
            TopicItem("Electromagnetism", "Electric Fields and Potential"),
        ]
        normalized = _normalize_semantic_result(
            {
                "topic_guess": {
                    "domain": "Electromagnetic Theory",
                    "topic": "Electric Field and Electric Potential",
                },
                "semantic_audit": {"is_correct": False},
                "experience_rules": [
                    {
                        "title": "检查电势关系",
                        "trigger": "解答混淆电场与电势时",
                        "check_logic": "核对电场是否等于电势的负梯度",
                    }
                ],
            },
            sample_id="new_1",
            topics=topics,
            max_rules_per_sample=2,
        )
        self.assertEqual(
            normalized["topic_guess"],
            {
                "domain": "Electromagnetism",
                "topic": "Electric Fields and Potential",
            },
        )
        self.assertEqual(normalized["status"], "ok")


if __name__ == "__main__":
    unittest.main()
