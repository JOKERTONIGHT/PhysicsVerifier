import datetime
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.rule_based_verifier import RuleBasedVerifier
from symbolic.experience_code_engine import ExperienceCodeEngine


@dataclass
class _HintSpecCompat:
    spec_id: str
    title: str
    description: str
    primitive: str
    params: Dict[str, Any]
    source_rule_id: str
    source_message_substring: str


class TopDownVerifier:
    def __init__(
        self,
        rules_catalog_path: str = "catalogs/rules_catalog_top_down.json",
        llm_model: str = "qwen3-30b-a3b",
        log_dir: str = "logs",
        results_dir: str = "results",
        enable_agentic_postcheck: bool = True,
        agentic_max_checks_per_sample: int = 2,
        enable_experience_pipeline: bool = True,
        experience_rules_path: str = "results/semantic_experience_distilled_300.json",
        unified_rules_path: Optional[str] = None,
        experience_code_manifest_path: str = "results/experience_symbolic_program_manifest_300.json",
        experience_code_module: str = "symbolic.generated_experience_checks",
    ):
        self.llm_model = llm_model
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self._unified_mode = False
        if unified_rules_path and Path(unified_rules_path).exists():
            self.rules_catalog_path = unified_rules_path
            self._unified_mode = True
        else:
            self.rules_catalog_path = rules_catalog_path

        self.catalog = self._load_catalog()
        self.topics = self._flatten_topics()

        self.rule_verifier = RuleBasedVerifier(
            llm_model=self.llm_model,
            rule_mode="srd",
            rule_translations_path="rule_translations.json",
        )
        self.rule_verifier.rule_translations = {}

        # Enforce one-pass architecture: no agentic/spec fallback.
        self.enable_agentic_postcheck = False
        self.agentic_max_checks_per_sample = int(agentic_max_checks_per_sample)

        # Only used when running with non-unified catalogs.
        self.enable_experience_pipeline = bool(enable_experience_pipeline) and not self._unified_mode
        self.experience_rules_path = str(experience_rules_path)
        self.experience_rules_index = self._load_experience_rules(self.experience_rules_path) if self.enable_experience_pipeline else {}

        self.experience_code_engine = ExperienceCodeEngine(
            manifest_path=experience_code_manifest_path,
            module_name=experience_code_module,
        )

        self.error_experiences: List[Dict[str, Any]] = []

    def _normalize_topic_key(self, domain: str, topic: str) -> str:
        d = str(domain or "Unknown").strip().lower()
        t_raw = str(topic or "Unknown").strip()
        t = t_raw.split("/")[-1].strip().lower() if "/" in t_raw else t_raw.lower()
        return f"{d}::{t}"

    def _load_experience_rules(self, path: str) -> Dict[str, List[Dict[str, Any]]]:
        idx: Dict[str, List[Dict[str, Any]]] = {}
        p = Path(path)
        if not p.exists():
            return idx
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return idx

        rules = data.get("rules") if isinstance(data, dict) else []
        if not isinstance(rules, list):
            return idx

        for rule in rules:
            if not isinstance(rule, dict):
                continue
            key = self._normalize_topic_key(rule.get("domain"), rule.get("topic"))
            idx.setdefault(key, []).append(rule)
        return idx

    def _get_experience_rules_for_topic(self, topic: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not topic:
            return []
        key = self._normalize_topic_key(topic.get("domain"), topic.get("name"))
        return list(self.experience_rules_index.get(key, []))

    def _load_catalog(self) -> Dict[str, Any]:
        with open(self.rules_catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _flatten_topics(self) -> List[Dict[str, Any]]:
        topics: List[Dict[str, Any]] = []
        for domain in self.catalog.get("domains", []):
            domain_name = domain.get("name")
            for topic in domain.get("topics", []):
                t = topic.copy()
                t["domain"] = domain_name
                topics.append(t)
        return topics

    @staticmethod
    def _build_srd_for_rule(rule: Dict[str, Any]) -> str:
        source = rule.get("source", "knowledge")
        if source == "experience_tagged":
            return str(rule.get("description") or "")
        if source == "experience":
            parts: List[str] = []
            if rule.get("title"):
                parts.append(f"Title: {rule['title']}")
            if rule.get("trigger"):
                parts.append(f"Trigger: {rule['trigger']}")
            if rule.get("check_logic"):
                parts.append(f"Check Logic: {rule['check_logic']}")
            return "\n".join(parts)
        return (
            f"Title: {rule.get('title')}\n"
            f"Description: {rule.get('description')}\n"
            f"Check Logic: {rule.get('check_logic')}"
        )

    @staticmethod
    def _build_experience_symbolic_spec_from_hint(
        *,
        rule_id: str,
        title: str,
        check_logic: str,
        symbolic_hint: Dict[str, Any],
    ) -> Optional[_HintSpecCompat]:
        """Compatibility helper kept for tests; not used by runtime pipeline."""
        primitive = str(symbolic_hint.get("primitive") or "none").strip()
        if primitive in {"", "none"}:
            return None

        canonical = str(symbolic_hint.get("canonical") or "").strip()
        required_symbols = [str(s) for s in (symbolic_hint.get("required_symbols") or []) if str(s).strip()]

        params: Dict[str, Any] = {}
        if primitive in {"equation_equivalence", "inequality_consistency"}:
            if not canonical:
                return None
            params = {
                "canonical_latex": [canonical],
                "required_symbols": required_symbols,
            }
        elif primitive == "formula_pattern":
            if not required_symbols:
                return None
            params = {
                "patterns": [{"all_tokens": required_symbols, "relation": "=" if "=" in canonical else None}],
                "required_symbols": required_symbols,
            }
        elif primitive == "power_law":
            if len(required_symbols) < 2:
                return None
            params = {
                "dependent_candidates": required_symbols[:1],
                "independent_candidates": required_symbols[1:],
                "expected_exponent": 1,
                "tolerance": 0.1,
            }
        else:
            return None

        return _HintSpecCompat(
            spec_id=f"unified_hint_{rule_id}",
            title=f"Symbolic hint: {title}",
            description=check_logic,
            primitive=primitive,
            params=params,
            source_rule_id=rule_id,
            source_message_substring=title[:200] if title else "",
        )

    def classify_topic(self, question: str) -> Optional[Dict[str, Any]]:
        topic_list_str = "\n".join([f"- {t['domain']}: {t['name']}" for t in self.topics])

        prompt = f"""
You are a physics expert. Classify the following physics problem into one of the provided domains and topics.
Return ONLY the JSON object with "domain" and "topic" keys.

Problem:
{question[:1000]}...

Available Topics:
{topic_list_str}

JSON Output:
"""
        response = self.rule_verifier._llm_json(
            system_prompt="You are a classifier.",
            user_prompt=prompt,
        )

        if isinstance(response, dict):
            domain = str(response.get("domain") or response.get("Domain") or "").strip()
            topic_name = str(
                response.get("topic")
                or response.get("Topic")
                or response.get("name")
                or ""
            ).strip()

            if topic_name:
                target_key = self._normalize_topic_key(domain, topic_name)

                for t in self.topics:
                    if self._normalize_topic_key(t.get("domain"), t.get("name")) == target_key:
                        return t

                topic_norm = self._normalize_topic_key("", topic_name).split("::", 1)[1]
                for t in self.topics:
                    t_norm = self._normalize_topic_key("", t.get("name")).split("::", 1)[1]
                    if t_norm == topic_norm:
                        return t

                topic_norm_compact = topic_norm.replace(" ", "")
                for t in self.topics:
                    t_norm = self._normalize_topic_key("", t.get("name")).split("::", 1)[1].replace(" ", "")
                    if topic_norm_compact and (topic_norm_compact in t_norm or t_norm in topic_norm_compact):
                        return t

        q_tokens = [w.lower() for w in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", question or "") if len(w) >= 3]
        if q_tokens:
            best_topic: Optional[Dict[str, Any]] = None
            best_score = 0
            for t in self.topics:
                label = f"{t.get('domain', '')} {t.get('name', '')}".lower()
                score = sum(1 for w in q_tokens if w in label)
                if score > best_score:
                    best_score = score
                    best_topic = t
            if best_topic is not None and best_score >= 2:
                return best_topic

        return None

    def verify(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get("question", "")
        topic = self.classify_topic(question)

        diagnostics: List[Dict[str, Any]] = []
        verifier_used = "top_down_rule_based"
        suppressed_diagnostics: List[Dict[str, Any]] = []
        experience_post_diagnostics: List[Dict[str, Any]] = []
        experience_code_post_diagnostics: List[Dict[str, Any]] = []

        if topic:
            print(f"Classified into: {topic['domain']} - {topic['name']}")
            rules = topic.get("rules", []) or []
            if rules:
                current_translations: Dict[str, Dict[str, str]] = {}
                rule_ids: List[str] = []
                for r in rules:
                    rid = r.get("id")
                    if not rid:
                        continue
                    rule_ids.append(rid)
                    current_translations[rid] = {"srd": self._build_srd_for_rule(r)}

                self.rule_verifier.rules_to_check = rule_ids
                self.rule_verifier.rule_translations = current_translations

                print(f"Running rule check with {len(rule_ids)} rules...")
                result = self.rule_verifier.analyze(sample)
                diagnostics = result.get("diagnostics", [])
                verifier_used = "unified_rule_based" if self._unified_mode else "top_down_rule_based"
        else:
            print("Could not classify topic or no topic found.")

        # Strict reconciliation: each retained diagnostic must be bound to experience code.
        if diagnostics:
            reconciled: List[Dict[str, Any]] = []
            for d in diagnostics:
                if not isinstance(d, dict):
                    continue
                rid = str(d.get("rule") or "").strip()
                if not rid:
                    continue

                if not (self.experience_code_engine.available and self.experience_code_engine.has_rule(rid)):
                    suppressed_diagnostics.append(
                        {
                            "reason": "missing_experience_code_binding",
                            "rule_id": rid,
                            "original_diagnostic": d,
                        }
                    )
                    continue

                code_res = self.experience_code_engine.run_rule(rid, sample) or {
                    "result": "inconclusive",
                    "message": "experience code unavailable",
                    "evidence": "",
                }
                experience_code_post_diagnostics.append(
                    {
                        "rule": f"experience_code::{rid}",
                        "rule_id": rid,
                        "result": code_res.get("result"),
                        "message": code_res.get("message"),
                        "evidence": code_res.get("evidence"),
                    }
                )

                status = str(code_res.get("result") or "inconclusive")
                if status == "pass":
                    suppressed_diagnostics.append(
                        {
                            "reason": "experience_code_refuted_original",
                            "rule_id": rid,
                            "original_diagnostic": d,
                        }
                    )
                    continue

                d["symbolic_reconciliation"] = {
                    "status": "supported_by_experience_code" if status == "fail" else "inconclusive_by_experience_code",
                    "spec_ids": [],
                }
                d["experience_code_reconciliation"] = {
                    "status": status,
                    "rule_id": rid,
                }
                reconciled.append(d)

            diagnostics = reconciled

        # Non-unified mode optional bottom-up additions, still code-only.
        if self.enable_experience_pipeline and topic:
            exp_rules = self._get_experience_rules_for_topic(topic)
            for rule in exp_rules[:40]:
                rule_id = str(rule.get("rule_id") or "exp_unknown")
                if not (self.experience_code_engine.available and self.experience_code_engine.has_rule(rule_id)):
                    continue

                code_res = self.experience_code_engine.run_rule(rule_id, sample) or {
                    "result": "inconclusive",
                    "message": "experience code unavailable",
                    "evidence": "",
                }
                experience_code_post_diagnostics.append(
                    {
                        "rule": f"experience_code::{rule_id}",
                        "rule_id": rule_id,
                        "result": code_res.get("result"),
                        "message": code_res.get("message"),
                        "evidence": code_res.get("evidence"),
                    }
                )

                if code_res.get("result") == "fail":
                    experience_post_diagnostics.append(
                        {
                            "severity": "warning",
                            "rule": f"experience::{rule_id}",
                            "symbol": None,
                            "message": f"Experience code check failed: {rule.get('title')}",
                            "evidence": str(code_res.get("evidence") or "")[:200],
                            "experience_code_reconciliation": {
                                "status": "supported",
                                "rule_id": rule_id,
                            },
                            "experience_rule": {
                                "title": rule.get("title"),
                                "trigger": rule.get("trigger"),
                                "check_logic": rule.get("check_logic"),
                                "error_type": rule.get("error_type"),
                            },
                        }
                    )

            diagnostics.extend(experience_post_diagnostics)

        return {
            "id": sample.get("id"),
            "topic": topic.get("name") if topic else None,
            "verifier": verifier_used,
            "unified_mode": self._unified_mode,
            "diagnostics": diagnostics,
            "symbolic_post_diagnostics": [],
            "experience_post_diagnostics": experience_post_diagnostics,
            "experience_symbolic_post_diagnostics": [],
            "experience_code_post_diagnostics": experience_code_post_diagnostics,
            "agentic": {
                "enabled": False,
                "actions": [],
                "generated_specs": [],
                "suppressed_diagnostics": suppressed_diagnostics,
            },
            "experience_pipeline": {
                "enabled": self.enable_experience_pipeline,
                "actions": [],
                "rules_path": self.experience_rules_path,
                "experience_code_engine": {
                    "enabled": self.experience_code_engine.available,
                },
            },
            "score": -1.0 * len(diagnostics),
        }

    def _record_error_experience(self, sample: Dict[str, Any], topic: Optional[Dict[str, Any]], errors: List[Dict[str, Any]]):
        experience = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sample_id": sample.get("id"),
            "topic": topic.get("name") if topic else "Unknown",
            "question_snippet": sample.get("question", "")[:200],
            "errors_found": errors,
        }
        self.error_experiences.append(experience)

        exp_file = self.results_dir / "error_experiences.json"
        all_exps: List[Dict[str, Any]] = []
        if exp_file.exists():
            try:
                all_exps = json.loads(exp_file.read_text(encoding="utf-8"))
            except Exception:
                all_exps = []

        all_exps.append(experience)
        exp_file.write_text(json.dumps(all_exps, indent=2, ensure_ascii=False), encoding="utf-8")

    def run_batch(self, samples: List[Dict[str, Any]]):
        results = []
        for s in samples:
            print(f"Verifying sample {s.get('id')}...")
            res = self.verify(s)
            results.append(res)
        return results


if __name__ == "__main__":
    import sys

    data_path = "data/evaluation_sample_30.json"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]

    try:
        with open(data_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
    except FileNotFoundError:
        print(f"File {data_path} not found.")
        sys.exit(1)

    verifier = TopDownVerifier()
    results = verifier.run_batch(samples)

    with open("results/top_down_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Done. Results saved to results/top_down_results.json")
