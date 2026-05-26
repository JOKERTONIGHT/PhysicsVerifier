from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from scripts.generate_cluster_proposals import (
    _assert_english_only_proposal,
    _build_embedding_topic_prompt_payload,
    _build_client,
    _build_distilled_auxiliary_index,
    _build_topic_prompt_payload,
    _collect_topic_candidates,
    _chat_json,
    _extract_json_object,
    generate_cluster_proposals_from_embedding_clusters,
)


TMP_ROOT = Path("results/test_tmp")
TMP_ROOT.mkdir(parents=True, exist_ok=True)


def _case_dir() -> Path:
    path = TMP_ROOT / f"cluster_proposal_test_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


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

    def test_extract_json_object_accepts_unclosed_fenced_json(self) -> None:
        data = _extract_json_object(
            """```json
{"topic_summary":"mechanics summary","should_add_clusters":true}"""
        )
        self.assertEqual(data["topic_summary"], "mechanics summary")
        self.assertTrue(data["should_add_clusters"])

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

    def test_chat_json_allows_cjk_generated_text_for_bilingual_rules(self) -> None:
        class _Message:
            content = '{"topic_summary":"中文 summary","should_add_clusters":false,"clusters":[]}'

        class _Choice:
            message = _Message()

        class _Completions:
            def create(self, **kwargs):
                return type("Response", (), {"choices": [_Choice()]})()

        client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()

        result = _chat_json(
            client,
            model="test-model",
            temperature=0.0,
            system_prompt="system",
            user_prompt="user",
        )

        self.assertEqual(result["topic_summary"], "中文 summary")

    def test_build_client_passes_request_timeout(self) -> None:
        import scripts.generate_cluster_proposals as module

        original_openai = module.OpenAI
        original_httpx = module.httpx

        class _Httpx:
            class Client:
                def __init__(self, **kwargs):
                    self.kwargs = kwargs

        captured = {}

        class _OpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        try:
            module.OpenAI = _OpenAI
            module.httpx = _Httpx
            _build_client(api_key="key", base_url="https://example.test", trust_env=True, request_timeout=123.0)
        finally:
            module.OpenAI = original_openai
            module.httpx = original_httpx

        self.assertEqual(captured["timeout"], 123.0)
        self.assertEqual(captured["http_client"].kwargs["timeout"], 123.0)
        self.assertTrue(captured["http_client"].kwargs["trust_env"])

    def test_embedding_cluster_payload_samples_rules_without_reassigning_membership(self) -> None:
        payload = _build_embedding_topic_prompt_payload(
            {
                "domain": "Mechanics",
                "topic": "Kinematics",
                "topic_key": "Mechanics::Kinematics",
                "rule_count": 3,
                "clusters": [
                    {
                        "cluster_id": "embedding_cluster_01",
                        "size": 2,
                        "rule_ids": ["r1", "r2"],
                        "representative_rules": [{"rule_id": "r1", "summary": "motion summary"}],
                    }
                ],
                "residual_rule_ids": ["r3"],
            },
            rule_index={
                "r1": {"title": "t1", "summary": "s1", "trigger": "tr1", "check_logic": "c1"},
                "r2": {"title": "t2", "summary": "s2", "trigger": "tr2", "check_logic": "c2"},
            },
            max_rules_per_cluster=1,
        )

        self.assertEqual(payload["embedding_clusters"][0]["source_cluster_id"], "embedding_cluster_01")
        self.assertEqual([item["rule_id"] for item in payload["embedding_clusters"][0]["sampled_rules"]], ["r1"])
        self.assertNotIn("candidate_rule_ids", payload["output_schema"]["clusters"][0])
        self.assertIn("source_cluster_id", payload["output_schema"]["clusters"][0])

    def test_generate_from_embedding_clusters_keeps_rule_membership_deterministic(self) -> None:
        class _Message:
            content = (
                '{"topic_summary":"Kinematics timing scenes","rationale":"embedding groups are fixed",'
                '"clusters":[{"source_cluster_id":"embedding_cluster_01","cluster_id":"timing_checks",'
                '"name":"Timing Checks","summary":"Checks timing relations.","description":"Time relation scenarios.",'
                '"scene_cues":["time interval"],"boundary_cues":["force dynamics"],"explore_cues":["piecewise motion"]}]}'
            )

        class _Choice:
            message = _Message()

        class _Completions:
            def create(self, **kwargs):
                return type("Response", (), {"choices": [_Choice()]})()

        client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
        result = generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters={
                "topics": [
                    {
                        "domain": "Mechanics",
                        "topic": "Kinematics",
                        "topic_key": "Mechanics::Kinematics",
                        "rule_count": 3,
                        "clusters": [{"cluster_id": "embedding_cluster_01", "rule_ids": ["r1", "r2"], "size": 2}],
                        "residual_rule_ids": ["r3"],
                    }
                ]
            },
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            max_output_tokens=2048,
        )

        proposal = result["proposals"][0]
        self.assertEqual(proposal["clusters"][0]["candidate_rule_ids"], ["r1", "r2"])
        self.assertEqual(proposal["residual_rule_ids"], ["r3"])
        self.assertEqual(result["metadata"]["generator"], "embedding_cluster_labeling_v1")

    def test_generate_from_embedding_clusters_records_cjk_warning_without_failure(self) -> None:
        class _Message:
            content = (
                '{"topic_summary":"中文场景","rationale":"fixed",'
                '"clusters":[{"source_cluster_id":"embedding_cluster_01","cluster_id":"gas_escape",'
                '"name":"Gas Escape","summary":"流出分子的平均能量为 2kT","description":"gas escape",'
                '"scene_cues":[],"boundary_cues":[],"explore_cues":[]}]}'
            )

        class _Choice:
            message = _Message()

        class _Completions:
            def create(self, **kwargs):
                return type("Response", (), {"choices": [_Choice()]})()

        client = type("Client", (), {"chat": type("Chat", (), {"completions": _Completions()})()})()
        result = generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters={
                "topics": [
                    {
                        "domain": "Thermodynamics",
                        "topic": "Kinetic Theory",
                        "topic_key": "Thermodynamics::Kinetic Theory",
                        "rule_count": 4,
                        "clusters": [{"cluster_id": "embedding_cluster_01", "rule_ids": ["r1", "r2"], "size": 2}],
                        "residual_rule_ids": [],
                    }
                ]
            },
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
        )

        self.assertTrue(result["proposals"][0]["contains_cjk_generated_text"])
        self.assertEqual(result["metadata"]["cjk_warning_count"], 1)

    def test_generate_from_embedding_clusters_saves_incremental_output_and_resumes(self) -> None:
        root = _case_dir()
        output_path = root / "cluster_proposals.json"

        class _Message:
            content = (
                '{"topic_summary":"summary","rationale":"fixed embedding clusters",'
                '"clusters":[{"source_cluster_id":"embedding_cluster_01","cluster_id":"timing_checks",'
                '"name":"Timing Checks","summary":"Checks timing.","description":"Timing scenarios.",'
                '"scene_cues":[],"boundary_cues":[],"explore_cues":[]}]}'
            )

        class _Choice:
            message = _Message()

        class _Completions:
            def __init__(self) -> None:
                self.calls = 0

            def create(self, **kwargs):
                self.calls += 1
                return type("Response", (), {"choices": [_Choice()]})()

        completions = _Completions()
        client = type("Client", (), {"chat": type("Chat", (), {"completions": completions})()})()
        embedding_clusters = {
            "topics": [
                {
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "topic_key": "Mechanics::Kinematics",
                    "rule_count": 3,
                    "clusters": [{"cluster_id": "embedding_cluster_01", "rule_ids": ["r1", "r2"], "size": 2}],
                    "residual_rule_ids": ["r3"],
                }
            ]
        }

        generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters=embedding_clusters,
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            output_path=output_path,
            resume=False,
        )
        generate_cluster_proposals_from_embedding_clusters(
            embedding_clusters=embedding_clusters,
            rule_input={"rules": [{"rule_id": "r1", "summary": "s1"}, {"rule_id": "r2", "summary": "s2"}]},
            client=client,
            model="test-model",
            temperature=0.0,
            max_topics=1,
            min_rule_count=1,
            max_rules_per_cluster=2,
            output_path=output_path,
            resume=True,
        )

        self.assertEqual(completions.calls, 1)
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
