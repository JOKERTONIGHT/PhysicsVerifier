from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

class UnifiedCatalogTests(unittest.TestCase):
    def test_unified_catalog_loads_both_sources(self) -> None:
        """Verify that a unified catalog file loads both knowledge and experience rules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "rules_unified.json"
            payload = {
                "metadata": {"version": "1.0", "total_rules": 3},
                "domains": [
                    {
                        "name": "Mechanics",
                        "topics": [
                            {
                                "name": "Kinematics",
                                "rules": [
                                    {
                                        "id": "kin_rule_01",
                                        "title": "Kinematic equation",
                                        "description": "v = u + at",
                                        "source": "knowledge",
                                        "source_file": "rules_catalog_top_down.json",
                                        "tags": {"domain": "Mechanics", "topic": "Kinematics"},
                                        "check_logic": "Verify v = u + at",
                                    },
                                    {
                                        "id": "exp_kin_001",
                                        "title": "速度方向判断",
                                        "description": "Tagged experience rule with long description.",
                                        "source": "experience_tagged",
                                        "source_file": "rules_300_tagged.json",
                                        "tags": {"domain": "Mechanics", "topic": "Kinematics"},
                                    },
                                    {
                                        "id": "exp_abc123",
                                        "title": "初速度校验",
                                        "description": "触发条件：涉及初速度为零时\n检查逻辑：验证是否正确设置v0=0",
                                        "source": "experience",
                                        "source_file": "semantic_experience_distilled_300.json",
                                        "tags": {"domain": "Mechanics", "topic": "Kinematics"},
                                        "check_logic": "验证是否正确设置v0=0",
                                        "trigger": "涉及初速度为零时",
                                        "symbolic_hint": {
                                            "primitive": "equation_equivalence",
                                            "canonical": "v = a * t",
                                            "required_symbols": ["v", "a", "t"],
                                        },
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
            catalog_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            from core.physics_rule_verifier import PhysicsRuleVerifier

            verifier = PhysicsRuleVerifier(
                llm_model=None,
                unified_rules_path=str(catalog_path),
            )

            self.assertTrue(verifier._unified_mode)
            self.assertEqual(len(verifier.topics), 1)

            topic = verifier.topics[0]
            self.assertEqual(topic["name"], "Kinematics")
            self.assertEqual(len(topic["rules"]), 3)

            sources = {r["source"] for r in topic["rules"]}
            self.assertEqual(sources, {"knowledge", "experience_tagged", "experience"})

    def test_unified_srd_construction(self) -> None:
        """Verify that _build_srd_for_rule constructs correct SRD for each source type."""
        from core.physics_rule_verifier import PhysicsRuleVerifier

        # Knowledge rule
        knowledge_rule = {
            "source": "knowledge",
            "title": "My Title",
            "description": "My Description",
            "check_logic": "My Check Logic",
        }
        srd = PhysicsRuleVerifier._build_srd_for_rule(knowledge_rule)
        self.assertIn("Title: My Title", srd)
        self.assertIn("Description: My Description", srd)
        self.assertIn("Check Logic: My Check Logic", srd)

        # Tagged experience rule: description IS the full SRD
        tagged_rule = {
            "source": "experience_tagged",
            "title": "Some Title",
            "description": "This is a very long detailed SRD that stands alone.",
            "check_logic": "This should not appear in SRD",
        }
        srd = PhysicsRuleVerifier._build_srd_for_rule(tagged_rule)
        self.assertEqual(srd, "This is a very long detailed SRD that stands alone.")
        self.assertNotIn("Title:", srd)

        # Distilled experience rule: trigger + check_logic
        distilled_rule = {
            "source": "experience",
            "title": "Distilled Title",
            "trigger": "When condition X",
            "check_logic": "Check Y equals Z",
        }
        srd = PhysicsRuleVerifier._build_srd_for_rule(distilled_rule)
        self.assertIn("Title: Distilled Title", srd)
        self.assertIn("Trigger: When condition X", srd)
        self.assertIn("Check Logic: Check Y equals Z", srd)

    def test_build_experience_symbolic_spec_from_hint(self) -> None:
        """Verify symbolic_hint conversion to GeneratedSymbolicCheckSpec."""
        from core.physics_rule_verifier import PhysicsRuleVerifier

        # equation_equivalence with valid hint
        spec = PhysicsRuleVerifier._build_experience_symbolic_spec_from_hint(
            rule_id="test_001",
            title="Test Rule",
            check_logic="Check something",
            symbolic_hint={
                "primitive": "equation_equivalence",
                "canonical": "v = u + at",
                "required_symbols": ["v", "u", "a", "t"],
            },
        )
        self.assertIsNotNone(spec)
        self.assertEqual(spec.spec_id, "unified_hint_test_001")
        self.assertEqual(spec.primitive, "equation_equivalence")
        self.assertIn("v = u + at", spec.params["canonical_latex"])

        # none primitive should return None
        spec = PhysicsRuleVerifier._build_experience_symbolic_spec_from_hint(
            rule_id="test_002",
            title="Test Rule 2",
            check_logic="Check other",
            symbolic_hint={"primitive": "none", "canonical": "", "required_symbols": []},
        )
        self.assertIsNone(spec)


if __name__ == "__main__":
    unittest.main()