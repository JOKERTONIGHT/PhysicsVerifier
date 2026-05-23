from __future__ import annotations

import unittest

from scripts.generate_cluster_proposals import (
    _assert_english_only_proposal,
    _build_distilled_auxiliary_index,
    _build_topic_prompt_payload,
    _collect_topic_candidates,
    _chat_json,
    _extract_json_object,
)


class GenerateClusterProposalTests(unittest.TestCase):
    def test_extract_json_object_accepts_plain_json(self) -> None:
        data = _extract_json_object('{"topic_summary":"x","should_add_clusters":true}')
        self.assertEqual(data["topic_summary"], "x")
        self.assertTrue(data["should_add_clusters"])

    def test_extract_json_object_accepts_fenced_json(self) -> None:
        data = _extract_json_object(
            """Here is the proposal:

```json
{"topic_summary":"mechanics summary","should_add_clusters":false}
```"""
        )
        self.assertEqual(data["topic_summary"], "mechanics summary")
        self.assertFalse(data["should_add_clusters"])

    def test_extract_json_object_accepts_loose_wrapped_json(self) -> None:
        data = _extract_json_object(
            'I think this is the right output: {"topic_summary":"wrapped","should_add_clusters":true}'
        )
        self.assertEqual(data["topic_summary"], "wrapped")
        self.assertTrue(data["should_add_clusters"])

    def test_english_only_proposal_rejects_cjk_content(self) -> None:
        with self.assertRaises(RuntimeError):
            _assert_english_only_proposal(
                {
                    "topic_summary": "mechanics summary",
                    "rationale": "ok",
                    "clusters": [
                        {
                            "cluster_id": "bad_cluster",
                            "name": "中文名称",
                            "summary": "summary",
                            "description": "description",
                            "includes": [],
                            "excludes": [],
                            "entry_cues": [],
                            "related_clusters": [],
                        }
                    ],
                }
            )

    def test_minimal_catalog_payload_omits_removed_runtime_fields_and_adds_auxiliary(self) -> None:
        auxiliary_index = _build_distilled_auxiliary_index(
            {
                "rules": [
                    {
                        "rule_id": "exp_a",
                        "auxiliary": {
                            "node_summary": "Magnetic force direction check.",
                            "scene_cues": ["charged particle in uniform B"],
                            "boundary_cues": ["electric force dominates"],
                            "explore_cues": ["circular motion coupling"],
                            "evidence_sample_ids": ["s1", "s2"],
                        },
                    }
                ]
            }
        )
        payload = _build_topic_prompt_payload(
            {
                "domain": "Electromagnetism",
                "topic": "Magnetic Fields and Lorentz Force",
                "rule_count": 1,
                "topic_obj": {
                    "summary": "Magnetic force and charged-particle motion.",
                    "includes": ["old include should not be present"],
                    "excludes": ["old exclude should not be present"],
                    "related_topics": ["old relation should not be present"],
                    "scenario_clusters": [],
                    "rules": [
                        {
                            "rule_id": "exp_a",
                            "title": "Lorentz direction",
                            "summary": "Check magnetic force direction.",
                            "trigger": "qvB setup",
                            "check_logic": "Use right-hand rule with charge sign.",
                            "error_type": "concept",
                            "symbolic_hint": {"primitive": "formula_pattern", "canonical": "F=qvB", "required_symbols": ["q"]},
                            "scope": "old scope",
                            "support": {"count": 2, "sample_ids": ["s1"]},
                        }
                    ],
                },
            },
            auxiliary_by_rule=auxiliary_index,
        )

        self.assertNotIn("topic_includes", payload)
        self.assertNotIn("topic_excludes", payload)
        self.assertNotIn("related_topics", payload)
        rule_payload = payload["rules"][0]
        self.assertNotIn("scope", rule_payload)
        self.assertNotIn("support_count", rule_payload)
        self.assertNotIn("sample_ids", rule_payload)
        self.assertEqual(rule_payload["auxiliary"]["node_summary"], "Magnetic force direction check.")
        self.assertEqual(rule_payload["auxiliary"]["scene_cues"], ["charged particle in uniform B"])
        self.assertIn("scene_cues", payload["output_schema"]["clusters"][0])
        self.assertNotIn("includes", payload["output_schema"]["clusters"][0])

    def test_collect_topic_candidates_prioritizes_high_rule_missing_cluster_topics(self) -> None:
        catalog = {
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {"name": "Has Cluster", "rules": [{"rule_id": "a"}] * 100, "scenario_clusters": [{"id": "c"}]},
                        {"name": "Small Missing", "rules": [{"rule_id": "b"}] * 5, "scenario_clusters": []},
                        {"name": "Large Missing", "rules": [{"rule_id": "c"}] * 80, "scenario_clusters": []},
                    ],
                }
            ]
        }

        candidates = _collect_topic_candidates(
            catalog,
            only_missing_clusters=True,
            domain_filters=set(),
            topic_filters=set(),
            max_topics=1,
            min_rule_count=10,
        )

        self.assertEqual([(item["domain"], item["topic"], item["rule_count"]) for item in candidates], [("Mechanics", "Large Missing", 80)])

    def test_chat_json_passes_explicit_max_tokens(self) -> None:
        class _Message:
            content = '{"topic_summary":"x","should_add_clusters":false,"clusters":[]}'

        class _Choice:
            message = _Message()

        class _Completions:
            def __init__(self) -> None:
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return type("Response", (), {"choices": [_Choice()]})()

        completions = _Completions()
        client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()

        result = _chat_json(
            client,
            model="test-model",
            temperature=0.0,
            system_prompt="system",
            user_prompt="user",
            max_output_tokens=8192,
        )

        self.assertFalse(result["should_add_clusters"])
        self.assertEqual(completions.kwargs["max_tokens"], 8192)


if __name__ == "__main__":
    unittest.main()
