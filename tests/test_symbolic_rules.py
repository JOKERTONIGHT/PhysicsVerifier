from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

from symbolic.experience_code_engine import ExperienceCodeEngine
from symbolic.match_utils import symbol_match_report


class SymbolicRulesRegressionTests(unittest.TestCase):
    def test_symbol_match_report_uses_aliases(self) -> None:
        report = symbol_match_report("tau = 2pi", ["τ", "π"], min_ratio=1.0)
        self.assertTrue(report["ok"])
        self.assertEqual(report["missing"], [])

    def test_experience_code_engine_load_and_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            module_name = "tmp_checks_mod_a"
            mod_path = tmp / f"{module_name}.py"
            mod_path.write_text(
                """
def check_rule(sample):
    pred = str(sample.get('prediction') or '')
    if 'bad' in pred:
        return {'result': 'fail', 'message': 'found bad', 'evidence': pred[:50]}
    return {'result': 'pass', 'message': 'ok', 'evidence': pred[:50]}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            manifest_path = tmp / "manifest.json"
            payload = {
                "checks": [
                    {
                        "rule_id": "exp_test_001",
                        "domain": "Mechanics",
                        "topic": "Kinematics",
                        "function_name": "check_rule",
                    }
                ]
            }
            manifest_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

            sys.path.insert(0, str(tmp))
            try:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                engine = ExperienceCodeEngine(
                    manifest_path=str(manifest_path),
                    module_name=module_name,
                )
                self.assertTrue(engine.available)
                self.assertTrue(engine.has_rule("exp_test_001"))

                fail_out = engine.run_rule("exp_test_001", {"prediction": "this is bad"})
                self.assertIsNotNone(fail_out)
                self.assertEqual(fail_out["result"], "fail")

                pass_out = engine.run_rule("exp_test_001", {"prediction": "looks good"})
                self.assertIsNotNone(pass_out)
                self.assertEqual(pass_out["result"], "pass")
            finally:
                if sys.path and sys.path[0] == str(tmp):
                    sys.path.pop(0)
                if module_name in sys.modules:
                    del sys.modules[module_name]

    def test_experience_code_engine_topic_listing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            module_name = "tmp_checks_mod_b"
            mod_path = tmp / f"{module_name}.py"
            mod_path.write_text(
                """
def fn_a(sample):
    return {'result': 'pass', 'message': 'ok', 'evidence': ''}

def fn_b(sample):
    return {'result': 'inconclusive', 'message': 'n/a', 'evidence': ''}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            manifest_path = tmp / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "checks": [
                            {
                                "rule_id": "exp_a",
                                "domain": "Mechanics",
                                "topic": "Kinematics",
                                "function_name": "fn_a",
                            },
                            {
                                "rule_id": "exp_b",
                                "domain": "Mechanics",
                                "topic": "Kinematics",
                                "function_name": "fn_b",
                            },
                        ]
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            sys.path.insert(0, str(tmp))
            try:
                if module_name in sys.modules:
                    del sys.modules[module_name]
                engine = ExperienceCodeEngine(
                    manifest_path=str(manifest_path),
                    module_name=module_name,
                )
                ids = sorted(engine.list_topic_rule_ids("Mechanics", "Kinematics"))
                self.assertEqual(ids, ["exp_a", "exp_b"])
            finally:
                if sys.path and sys.path[0] == str(tmp):
                    sys.path.pop(0)
                if module_name in sys.modules:
                    del sys.modules[module_name]


if __name__ == "__main__":
    unittest.main()
