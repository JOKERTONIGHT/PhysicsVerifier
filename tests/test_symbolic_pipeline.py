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


if __name__ == "__main__":
    unittest.main()