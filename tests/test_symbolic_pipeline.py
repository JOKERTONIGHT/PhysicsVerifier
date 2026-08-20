from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class UnifiedCatalogTests(unittest.TestCase):
    def test_unified_catalog_loads_both_sources(self) -> None:
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
        from core.physics_rule_verifier import PhysicsRuleVerifier

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

        tagged_rule = {
            "source": "experience_tagged",
            "title": "Some Title",
            "description": "This is a very long detailed SRD that stands alone.",
            "check_logic": "This should not appear in SRD",
        }
        srd = PhysicsRuleVerifier._build_srd_for_rule(tagged_rule)
        self.assertEqual(srd, "This is a very long detailed SRD that stands alone.")
        self.assertNotIn("Title:", srd)

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


class ExperienceCodeSymbolicCheckTests(unittest.TestCase):
    """Verify that the verifier wires diagnostics through the experience-code engine."""

    def _build_verifier_with_temp_engine(self, *, fail_rule_id: str, pass_rule_id: str):
        from core.physics_rule_verifier import PhysicsRuleVerifier

        tmpdir = tempfile.mkdtemp(prefix="exp_code_test_")
        tmp = Path(tmpdir)
        module_name = "tmp_e2e_exp_checks_mod"
        mod_path = tmp / f"{module_name}.py"
        mod_path.write_text(
            "def check_fail(sample):\n"
            "    return {'result': 'fail', 'message': 'sym fail',"
            "            'evidence': 'velocity and acceleration'}\n"
            "\n"
            "def check_pass(sample):\n"
            "    return {'result': 'pass', 'message': 'sym pass',"
            "            'evidence': 'velocity and acceleration'}\n",
            encoding="utf-8",
        )
        manifest = {
            "checks": [
                {
                    "rule_id": fail_rule_id,
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "function_name": "check_fail",
                },
                {
                    "rule_id": pass_rule_id,
                    "domain": "Mechanics",
                    "topic": "Kinematics",
                    "function_name": "check_pass",
                },
            ]
        }
        manifest_path = tmp / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")

        catalog_path = tmp / "rules_unified.json"
        catalog = {
            "metadata": {"version": "2.0", "catalog_type": "unified_rules_v2"},
            "domains": [
                {
                    "name": "Mechanics",
                    "topics": [
                        {
                            "name": "Kinematics",
                            "rules": [
                                {
                                    "rule_id": fail_rule_id,
                                    "title": "Fail rule",
                                    "trigger": "velocity acceleration",
                                    "check_logic": "displacement",
                                    "error_type": "logic",
                                    "scope": "domain",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["v", "t"]},
                                    "support": {"count": 5, "sample_ids": ["1"]},
                                    "match_features": {
                                        "trigger_keywords": ["velocity", "acceleration"],
                                        "object_keywords": ["displacement"],
                                        "required_symbols": ["v", "t"],
                                        "primitive": "none",
                                    },
                                },
                                {
                                    "rule_id": pass_rule_id,
                                    "title": "Pass rule",
                                    "trigger": "velocity acceleration",
                                    "check_logic": "displacement",
                                    "error_type": "logic",
                                    "scope": "domain",
                                    "symbolic_hint": {"primitive": "none", "canonical": "", "required_symbols": ["v", "t"]},
                                    "support": {"count": 4, "sample_ids": ["2"]},
                                    "match_features": {
                                        "trigger_keywords": ["velocity", "acceleration"],
                                        "object_keywords": ["displacement"],
                                        "required_symbols": ["v", "t"],
                                        "primitive": "none",
                                    },
                                },
                            ],
                            "knowledge_reference": {"rule_ids": ["k1"], "keywords": ["velocity", "displacement"]},
                            "tagged_reference": {"source_ids": [], "titles": [], "aliases": [], "keywords": []},
                            "retrieval_hints": {
                                "scene_keywords": ["velocity"],
                                "topic_keywords": ["velocity", "displacement"],
                                "required_symbols": ["v", "t"],
                            },
                            "clusters": [],
                        }
                    ],
                }
            ],
        }
        catalog_path.write_text(json.dumps(catalog, ensure_ascii=False), encoding="utf-8")

        sys.path.insert(0, str(tmp))
        if module_name in sys.modules:
            del sys.modules[module_name]

        verifier = PhysicsRuleVerifier(
            llm_model=None,
            unified_rules_path=str(catalog_path),
            unified_retrieval_mode="lexical",
            experience_code_manifest_path=str(manifest_path),
            experience_code_module=module_name,
            log_dir=str(tmp / "logs"),
            results_dir=str(tmp / "results"),
        )
        return verifier, tmp, module_name

    def test_fail_marks_diagnostic_supported_and_pass_suppresses(self) -> None:
        fail_rule_id = "exp_test_fail_001"
        pass_rule_id = "exp_test_pass_001"
        verifier, tmp, module_name = self._build_verifier_with_temp_engine(
            fail_rule_id=fail_rule_id,
            pass_rule_id=pass_rule_id,
        )
        try:
            self.assertTrue(verifier.experience_code_engine.available)

            cached_diags = [
                {
                    "severity": "error",
                    "rule": fail_rule_id,
                    "symbol": None,
                    "message": "LLM critique reinforced by code",
                    "evidence": {"quote": "v + a t mismatch", "location": {"locatable_valid": True, "paragraph_index": 0}},
                },
                {
                    "severity": "error",
                    "rule": pass_rule_id,
                    "symbol": None,
                    "message": "LLM critique refuted by code",
                    "evidence": {"quote": "displacement", "location": {"locatable_valid": True, "paragraph_index": 1}},
                },
            ]
            verifier.semantic_checker.analyze = lambda sample: {"diagnostics": [dict(d) for d in cached_diags]}

            sample = {
                "id": "sample_exp_code",
                "question": "A particle moves with velocity and acceleration. Find displacement as a function of time.",
                "prediction": "Some prediction with velocity discussion.",
                "answer": "",
            }
            result = verifier.verify(sample)

            kept_rules = [d.get("rule") for d in result["diagnostics"]]
            self.assertIn(fail_rule_id, kept_rules)
            self.assertNotIn(pass_rule_id, kept_rules)

            symbolic_results = {(c.get("rule_id"), c.get("result")) for c in result["experience_code_post_diagnostics"]}
            self.assertIn((fail_rule_id, "fail"), symbolic_results)
            self.assertIn((pass_rule_id, "pass"), symbolic_results)

            recon_status = next(
                (
                    d.get("symbolic_reconciliation", {}).get("status")
                    for d in result["diagnostics"]
                    if d.get("rule") == fail_rule_id
                ),
                None,
            )
            self.assertEqual(recon_status, "supported")

            agentic = result.get("agentic") or {}
            suppressed = agentic.get("suppressed_diagnostics") or []
            suppressed_rules = [
                str((s.get("original_diagnostic") or {}).get("rule") or "")
                for s in suppressed
            ]
            self.assertIn(pass_rule_id, suppressed_rules)
        finally:
            if sys.path and sys.path[0] == str(tmp):
                sys.path.pop(0)
            if module_name in sys.modules:
                del sys.modules[module_name]

    def test_bottom_up_emits_diagnostic_for_topic_rule_without_llm_match(self) -> None:
        fail_rule_id = "exp_test_bu_fail_001"
        pass_rule_id = "exp_test_bu_pass_001"
        verifier, tmp, module_name = self._build_verifier_with_temp_engine(
            fail_rule_id=fail_rule_id,
            pass_rule_id=pass_rule_id,
        )
        try:
            verifier.semantic_checker.analyze = lambda sample: {"diagnostics": []}

            sample = {
                "id": "sample_bottom_up",
                "question": "A particle moves with velocity and acceleration. Find displacement as a function of time.",
                "prediction": "Use velocity and acceleration over time.",
                "answer": "",
            }
            result = verifier.verify(sample)

            new_rules = [d.get("rule") for d in result["diagnostics"]]
            self.assertIn(f"experience_code::{fail_rule_id}", new_rules)
            self.assertNotIn(f"experience_code::{pass_rule_id}", new_rules)

            results_by_rule = {
                c.get("rule_id"): c.get("result")
                for c in result["experience_code_post_diagnostics"]
            }
            self.assertEqual(results_by_rule.get(fail_rule_id), "fail")
            self.assertEqual(results_by_rule.get(pass_rule_id), "pass")
        finally:
            if sys.path and sys.path[0] == str(tmp):
                sys.path.pop(0)
            if module_name in sys.modules:
                del sys.modules[module_name]


if __name__ == "__main__":
    unittest.main()
