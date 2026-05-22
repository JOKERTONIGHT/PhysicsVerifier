from __future__ import annotations

import unittest

from scripts.run_semantic_experience import _build_distilled_library, _resume_done_map, _semantic_prompt


class SemanticExperienceAuxiliaryTests(unittest.TestCase):
    def test_distilled_rules_aggregate_navigation_auxiliary(self) -> None:
        samples_payload = [
            {
                "sample_id": "s1",
                "topic_guess": {"domain": "Mechanics", "topic": "Gravitation"},
                "experience_rules": [
                    {
                        "title": "Orbital energy check",
                        "trigger": "satellite drag changes orbital radius",
                        "check_logic": "verify orbital energy-radius relation",
                        "error_type": "logic",
                        "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["r"]},
                        "auxiliary": {
                            "node_summary": "Satellite orbital decay under drag.",
                            "scene_cues": ["satellite drag", "orbital radius"],
                            "boundary_cues": ["binary-star reduced mass"],
                            "explore_cues": ["energy accounting versus force modeling"],
                        },
                    }
                ],
            },
            {
                "sample_id": "s2",
                "topic_guess": {"domain": "Mechanics", "topic": "Gravitation"},
                "experience_rules": [
                    {
                        "title": "Orbital energy check",
                        "trigger": "satellite drag changes orbital radius",
                        "check_logic": "verify orbital energy-radius relation",
                        "error_type": "logic",
                        "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["r"]},
                        "auxiliary": {
                            "node_summary": "Orbital decay by drag.",
                            "scene_cues": ["satellite drag", "period change"],
                            "boundary_cues": ["binary-star reduced mass"],
                            "explore_cues": ["drag work determines energy loss"],
                        },
                    }
                ],
            },
        ]

        distilled = _build_distilled_library(samples_payload, min_count=1)

        self.assertEqual(distilled["summary"]["total_distilled_rules"], 1)
        auxiliary = distilled["rules"][0]["auxiliary"]
        self.assertEqual(auxiliary["node_summary"], "Orbital decay by drag.")
        self.assertEqual(auxiliary["scene_cues"], ["satellite drag", "orbital radius", "period change"])
        self.assertEqual(auxiliary["boundary_cues"], ["binary-star reduced mass"])
        self.assertEqual(
            auxiliary["explore_cues"],
            ["energy accounting versus force modeling", "drag work determines energy loss"],
        )
        self.assertEqual(auxiliary["evidence_sample_ids"], ["s1", "s2"])

    def test_old_semantic_output_distills_with_empty_auxiliary(self) -> None:
        samples_payload = [
            {
                "sample_id": "old1",
                "topic_guess": {"domain": "Mechanics", "topic": "Kinematics"},
                "experience_rules": [
                    {
                        "title": "Timing check",
                        "trigger": "time interval relation",
                        "check_logic": "verify displacement over time",
                        "error_type": "logic",
                        "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": []},
                    }
                ],
            }
        ]

        distilled = _build_distilled_library(samples_payload, min_count=1)

        self.assertEqual(
            distilled["rules"][0]["auxiliary"],
            {
                "node_summary": "",
                "scene_cues": [],
                "boundary_cues": [],
                "explore_cues": [],
                "evidence_sample_ids": [],
            },
        )

    def test_prompt_requests_navigation_auxiliary_schema(self) -> None:
        _, user_prompt = _semantic_prompt(
            sample={"id": "x", "question": "q", "prediction": "p", "answer": "a"},
            topics_block="- Mechanics / Kinematics",
            max_rules_per_sample=2,
        )

        self.assertIn('"auxiliary"', user_prompt)
        self.assertIn('"node_summary"', user_prompt)
        self.assertIn('"scene_cues"', user_prompt)
        self.assertIn('"boundary_cues"', user_prompt)
        self.assertIn('"explore_cues"', user_prompt)
        self.assertIn("真实题目", user_prompt)

    def test_resume_done_map_excludes_llm_failure_placeholders(self) -> None:
        existing_payload = {
            "samples": [
                {
                    "sample_id": "ok1",
                    "topic_guess": {"domain": "Mechanics", "topic": "Kinematics"},
                    "semantic_audit": {"summary": "valid", "key_errors": []},
                    "experience_rules": [{"title": "Timing"}],
                },
                {
                    "sample_id": "bad1",
                    "topic_guess": {"domain": "Unknown", "topic": "Unknown"},
                    "semantic_audit": {
                        "summary": "LLM调用失败，已记录重试占位。",
                        "key_errors": [{"message": "LLM调用失败", "evidence": "quota"}],
                    },
                    "experience_rules": [],
                },
            ]
        }

        done_map = _resume_done_map(existing_payload)

        self.assertIn("ok1", done_map)
        self.assertNotIn("bad1", done_map)


if __name__ == "__main__":
    unittest.main()
