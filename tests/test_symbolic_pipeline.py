from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.rule_based_verifier import RuleBasedVerifier
from rules.base import RuleContext
from rules.symbolic_checks import GeneratedSymbolicCheckExecutor, GeneratedSymbolicCheckSpec
from symbolic.spec_synthesis import RuleSymbolicSpecSynthesizer
from symbolic.symbolic_catalog import SymbolicCatalog


def build_context(text: str) -> RuleContext:
    verifier = RuleBasedVerifier(llm_model=None)
    parsed = verifier._extract_symbols_and_formulas(text)
    graph = verifier._build_symbol_graph(parsed["lines"], parsed["symbols"], parsed["formulas"])
    return RuleContext(
        sample_id="test",
        dataset_key=None,
        text_all=text,
        lines=parsed["lines"],
        symbols=parsed["symbols"],
        formulas_raw=parsed["formulas"],
        graph=graph,
        snippets={},
        sym_stats={},
        precondition_cues=[],
    )


class SymbolicPipelineTests(unittest.TestCase):
    def test_inequality_consistency_passes(self) -> None:
        ctx = build_context("Given the relativistic constraint, we must keep v < c.")
        executor = GeneratedSymbolicCheckExecutor()
        spec = GeneratedSymbolicCheckSpec(
            spec_id="ineq_speed_limit",
            title="Speed limit",
            description="",
            primitive="inequality_consistency",
            params={"canonical_latex": ["v < c"], "required_symbols": ["v", "c"]},
        )

        result = executor.run(ctx, [spec])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbolic_result"], "pass")

    def test_formula_pattern_fallback_handles_faraday_form(self) -> None:
        ctx = build_context(r"\oint \mathbf{E} \cdot d\mathbf{l} = -\frac{d\Phi_B}{dt}")
        executor = GeneratedSymbolicCheckExecutor()
        spec = GeneratedSymbolicCheckSpec(
            spec_id="faraday_pattern",
            title="Faraday integral law",
            description="",
            primitive="equation_equivalence",
            params={
                "canonical_latex": [r"\oint \mathbf{E} \cdot d\mathbf{l} = -\frac{d\Phi_B}{dt}"],
                "required_symbols": ["E", "Phi_B", "dt"],
                "allow_scalar_multiple": False,
            },
        )

        result = executor.run(ctx, [spec])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["symbolic_result"], "pass")

    def test_rule_text_synthesis_creates_equation_and_inequality_specs(self) -> None:
        synthesizer = RuleSymbolicSpecSynthesizer()
        topic = {
            "name": "Special Relativity",
            "rules": [
                {
                    "id": "velocity_limit_check_03",
                    "title": "Verify velocity bound",
                    "description": "The speed must remain less than c.",
                    "check_logic": "Ensure the derivation states v < c and uses \\gamma = 1 / \\sqrt{1 - v^2/c^2} correctly.",
                }
            ],
        }

        specs = synthesizer.synthesize_topic("Modern Physics", topic)
        derived = specs["velocity_limit_check_03"]
        primitives = {spec.primitive for spec in derived}
        self.assertIn("inequality_consistency", primitives)
        self.assertIn("equation_equivalence", primitives)

    def test_symbolic_catalog_requires_rule_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "symbolic_catalog.json"
            payload = {
                "domains": [
                    {
                        "name": "Mechanics",
                        "topics": [
                            {
                                "name": "Kinematics",
                                "checks": [
                                    {
                                        "spec_id": "kin_eq",
                                        "title": "Kinematics equation",
                                        "description": "",
                                        "primitive": "equation_equivalence",
                                        "params": {"canonical_latex": ["v = u + at"], "required_symbols": ["v", "u", "a", "t"]},
                                        "match_rule_ids": ["kin_rule"],
                                        "match_keywords": ["kinematic equation"],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
            catalog_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            catalog = SymbolicCatalog(path=str(catalog_path))

            matches = catalog.find_applicable(
                domain="Mechanics",
                topic="Kinematics",
                diagnostic={
                    "rule": "different_rule",
                    "message": "The kinematic equation is inconsistent.",
                    "evidence": {"quote": "v = u + at"},
                },
            )

            self.assertEqual(matches, [])


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

            from core.top_down_verifier import TopDownVerifier

            verifier = TopDownVerifier(
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
        from core.top_down_verifier import TopDownVerifier

        # Knowledge rule
        knowledge_rule = {
            "source": "knowledge",
            "title": "My Title",
            "description": "My Description",
            "check_logic": "My Check Logic",
        }
        srd = TopDownVerifier._build_srd_for_rule(knowledge_rule)
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
        srd = TopDownVerifier._build_srd_for_rule(tagged_rule)
        self.assertEqual(srd, "This is a very long detailed SRD that stands alone.")
        self.assertNotIn("Title:", srd)

        # Distilled experience rule: trigger + check_logic
        distilled_rule = {
            "source": "experience",
            "title": "Distilled Title",
            "trigger": "When condition X",
            "check_logic": "Check Y equals Z",
        }
        srd = TopDownVerifier._build_srd_for_rule(distilled_rule)
        self.assertIn("Title: Distilled Title", srd)
        self.assertIn("Trigger: When condition X", srd)
        self.assertIn("Check Logic: Check Y equals Z", srd)

    def test_build_experience_symbolic_spec_from_hint(self) -> None:
        """Verify symbolic_hint conversion to GeneratedSymbolicCheckSpec."""
        from core.top_down_verifier import TopDownVerifier

        # equation_equivalence with valid hint
        spec = TopDownVerifier._build_experience_symbolic_spec_from_hint(
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
        spec = TopDownVerifier._build_experience_symbolic_spec_from_hint(
            rule_id="test_002",
            title="Test Rule 2",
            check_logic="Check other",
            symbolic_hint={"primitive": "none", "canonical": "", "required_symbols": []},
        )
        self.assertIsNone(spec)


if __name__ == "__main__":
    unittest.main()