from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from rules.symbolic_checks import GeneratedSymbolicCheckExecutor, GeneratedSymbolicCheckSpec
from symbolic.match_utils import symbol_match_report
from symbolic.symbolic_catalog import SymbolicCatalog

from tests.test_symbolic_pipeline import build_context


class SymbolicRulesRegressionTests(unittest.TestCase):
    def test_symbol_match_report_uses_aliases(self) -> None:
        report = symbol_match_report("tau = 2pi", ["τ", "π"], min_ratio=1.0)
        self.assertTrue(report["ok"])
        self.assertEqual(report["missing"], [])

    def test_equation_equivalence_soft_required_symbol_gate(self) -> None:
        ctx = build_context("Given relation: I_avg = I_peak/pi.")
        executor = GeneratedSymbolicCheckExecutor()
        spec = GeneratedSymbolicCheckSpec(
            spec_id="avg_current_soft_gate",
            title="Average current relation",
            description="",
            primitive="equation_equivalence",
            params={
                "canonical_latex": ["I_{avg}=I_{peak}/\\pi"],
                "required_symbols": ["I_avg", "I_peak", "I_rms"],
                "required_symbol_min_ratio": 0.66,
                "allow_scalar_multiple": False,
            },
        )

        result = executor.run(ctx, [spec])
        self.assertEqual(len(result), 1)
        self.assertIn(result[0]["symbolic_result"], {"pass", "fail"})

    def test_catalog_find_applicable_uses_topic_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            catalog_path = Path(tmpdir) / "symbolic_catalog.json"
            payload = {
                "domains": [
                    {
                        "name": "Electromagnetism",
                        "topics": [
                            {
                                "name": "Circuit",
                                "checks": [
                                    {
                                        "spec_id": "circuit_rule",
                                        "title": "Circuit average current",
                                        "description": "",
                                        "primitive": "equation_equivalence",
                                        "params": {
                                            "canonical_latex": ["I_{avg}=I_{peak}/\\pi"],
                                            "required_symbols": ["I_avg", "I_peak", "I_rms"],
                                            "required_symbol_min_ratio": 0.66,
                                        },
                                        "match_rule_ids": ["avg_current_rule"],
                                        "match_keywords": ["average current"],
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
                domain="Electromagnetism",
                topic="Circuit",
                diagnostic={
                    "rule": "avg_current_rule",
                    "message": "average current relation I_avg = I_peak/pi is missing",
                    "evidence": {"quote": "I_avg = I_peak / pi"},
                },
            )

            self.assertEqual(len(matches), 1)
            self.assertEqual(matches[0].spec_id, "circuit_rule")


if __name__ == "__main__":
    unittest.main()
