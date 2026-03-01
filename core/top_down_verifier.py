import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.rule_based_verifier import RuleBasedVerifier, RuleContext
from rules.symbolic_checks import (
    GeneratedSymbolicCheckExecutor,
    GeneratedSymbolicCheckRegistry,
    GeneratedSymbolicCheckSpec,
    catalog_spec_to_generated,
)
from symbolic.symbolic_catalog import SymbolicCatalog, SymbolicCheckSpec

class TopDownVerifier:
    def __init__(
        self,
        rules_catalog_path: str = "catalogs/rules_catalog_top_down.json",
        llm_model: str = "qwen3-30b-a3b",
        log_dir: str = "logs",
        results_dir: str = "results",
        enable_agentic_postcheck: bool = True,
        agentic_max_checks_per_sample: int = 2,
    ):
        self.rules_catalog_path = rules_catalog_path
        self.llm_model = llm_model
        self.log_dir = Path(log_dir)
        self.results_dir = Path(results_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.catalog = self._load_catalog()
        self.topics = self._flatten_topics()
        
        # Initialize the base verifiers
        # We will dynamically update rules for the rule-based verifier
        self.rule_verifier = RuleBasedVerifier(
            llm_model=self.llm_model,
            rule_mode='srd', # We will inject SRDs dynamically
            rule_translations_path="rule_translations.json" # Dummy path, we'll overwrite
        )
        # Clear initial translations as we will set them per request
        self.rule_verifier.rule_translations = {} 
        
        self.enable_agentic_postcheck = bool(enable_agentic_postcheck)
        self.agentic_max_checks_per_sample = int(agentic_max_checks_per_sample)
        # Backward compatible registry (results/*) is kept for audit logs, but the source of truth is catalogs/symbolic_catalog.json
        self.symbolic_registry = GeneratedSymbolicCheckRegistry(path=str(self.results_dir / "agentic_symbolic_checks.json"))
        self.symbolic_catalog = SymbolicCatalog(path="catalogs/symbolic_catalog.json")
        self.symbolic_executor = GeneratedSymbolicCheckExecutor()

        self.error_experiences = []

    def _load_catalog(self) -> Dict:
        with open(self.rules_catalog_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _flatten_topics(self) -> List[Dict]:
        topics = []
        for domain in self.catalog.get("domains", []):
            domain_name = domain.get("name")
            for topic in domain.get("topics", []):
                t = topic.copy()
                t["domain"] = domain_name
                topics.append(t)
        return topics

    def classify_topic(self, question: str) -> Optional[Dict]:
        # Use LLM to classify the question into one of the topics
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
        # We can use the rule_verifier's LLM method for this
        response = self.rule_verifier._llm_json(
            system_prompt="You are a classifier.",
            user_prompt=prompt
        )
        
        if isinstance(response, dict):
            domain = response.get("domain")
            topic_name = response.get("topic")
            
            # Find the matching topic object
            for t in self.topics:
                if t["name"] == topic_name and (not domain or t["domain"] == domain):
                    return t
            
            # Fallback: try fuzzy match or just return None
            # For now, strict match on topic name
            for t in self.topics:
                if t["name"] == topic_name:
                    return t
                    
        return None

    def verify(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        question = sample.get("question", "")
        
        # 1. Classify
        topic = self.classify_topic(question)
        
        diagnostics: List[Dict[str, Any]] = []
        used_rules: List[str] = []
        verifier_used = "top_down_rule_based"
        agentic_actions: List[Dict[str, Any]] = []
        symbolic_post_diagnostics: List[Dict[str, Any]] = []
        generated_specs: List[Dict[str, Any]] = []
        suppressed_diagnostics: List[Dict[str, Any]] = []
        
        if topic:
            print(f"Classified into: {topic['domain']} - {topic['name']}")
            rules = topic.get("rules", [])
            
            if rules:
                # 2. Prepare Rule Verifier
                # Convert catalog rules to the format expected by RuleBasedVerifier
                # We use 'check_logic' as the SRD/description for the LLM
                current_translations = {}
                rule_ids = []
                for r in rules:
                    rid = r.get("id")
                    if not rid: continue
                    rule_ids.append(rid)
                    current_translations[rid] = {
                        "srd": f"Title: {r.get('title')}\nDescription: {r.get('description')}\nCheck Logic: {r.get('check_logic')}"
                    }
                
                self.rule_verifier.rules_to_check = rule_ids
                self.rule_verifier.rule_translations = current_translations
                
                # 3. Run Rule Check
                print(f"Running rule check with {len(rule_ids)} rules...")
                result = self.rule_verifier.analyze(sample)
                diagnostics = result.get("diagnostics", [])
                used_rules = rule_ids
                verifier_used = "top_down_rule_based"
        else:
            print("Could not classify topic or no topic found.")

        # 4. Agentic post-check: for each diagnostic, decide whether to build a symbolic cross-check spec.
        if self.enable_agentic_postcheck and diagnostics and self.rule_verifier._llm_available():
            text_all = "\n".join([
                sample.get("question", ""),
                sample.get("context", ""),
                sample.get("prediction", ""),
            ])
            parsed = self.rule_verifier._extract_symbols_and_formulas(text_all)
            graph = self.rule_verifier._build_symbol_graph(parsed["lines"], parsed["symbols"], parsed["formulas"])
            ctx = RuleContext(
                sample_id=str(sample.get("id")),
                dataset_key=None,
                text_all=text_all,
                lines=parsed["lines"],
                symbols=parsed["symbols"],
                formulas_raw=parsed["formulas"],
                graph=graph,
                snippets={},
                sym_stats={},
                precondition_cues=[],
            )

            checks_used = 0
            for d in diagnostics:
                if checks_used >= self.agentic_max_checks_per_sample:
                    break
                if not isinstance(d, dict):
                    continue
                rid = d.get("rule")
                if not rid:
                    continue

                # 4.1 Try to find an existing symbolic check from catalog first.
                domain_name = topic.get("domain") if topic else "Unknown"
                topic_name = topic.get("name") if topic else "Unknown"
                existing = self.symbolic_catalog.find_applicable(domain=domain_name, topic=topic_name, diagnostic=d)
                if existing:
                    # Convert to executor specs and run immediately.
                    specs = [catalog_spec_to_generated(s) for s in existing]
                    spec_ids = [s.spec_id for s in specs]
                    d.setdefault("symbolic_cross_checks", []).extend(spec_ids)
                    symbolic_post_diagnostics.extend(self.symbolic_executor.run(ctx, specs))
                    agentic_actions.append({
                        "need": False,
                        "reason": "used_existing_symbolic_check",
                        "spec": [s.__dict__ for s in existing],
                        "diagnostic_rule": rid,
                    })
                    checks_used += 1
                    continue

                srd = None
                if topic:
                    for r in (topic.get("rules", []) or []):
                        if r.get("id") == rid:
                            srd = f"Title: {r.get('title')}\nDescription: {r.get('description')}\nCheck Logic: {r.get('check_logic')}"
                            break
                if not srd:
                    continue

                agent_payload = self._agentic_decide_symbolic_check(srd=srd, diagnostic=d)
                agentic_actions.append(agent_payload)

                spec_dict = agent_payload.get("spec") if isinstance(agent_payload, dict) else None
                need = bool(agent_payload.get("need")) if isinstance(agent_payload, dict) else False
                if not need or not isinstance(spec_dict, dict):
                    continue

                try:
                    spec_id = str(spec_dict.get("spec_id"))
                    spec = GeneratedSymbolicCheckSpec(
                        spec_id=spec_id,
                        title=str(spec_dict.get("title")),
                        description=str(spec_dict.get("description")),
                        primitive=str(spec_dict.get("primitive")),
                        params=dict(spec_dict.get("params") or {}),
                        source_rule_id=rid,
                        source_message_substring=str(d.get("message") or "")[:200],
                    )
                except Exception:
                    continue

                # Persist into domain/topic structured catalog
                dom = topic.get("domain") if topic else "Unknown"
                tname = topic.get("name") if topic else "Unknown"
                self.symbolic_catalog.upsert_check(
                    domain=str(dom),
                    topic=str(tname),
                    spec=SymbolicCheckSpec(
                        spec_id=spec.spec_id,
                        title=spec.title,
                        description=spec.description,
                        primitive=spec.primitive,
                        params=spec.params,
                        match_rule_ids=[rid],
                        match_keywords=[str(d.get("message") or "")[:60]],
                    ),
                )

                self.symbolic_registry.upsert(spec)
                generated_specs.append(spec.__dict__)
                d.setdefault("symbolic_cross_checks", []).append(spec.spec_id)
                checks_used += 1

            # Execute newly generated specs (avoid re-running the entire registry every time)
            if generated_specs:
                specs_to_run = [GeneratedSymbolicCheckSpec(**s) for s in generated_specs]
                symbolic_post_diagnostics.extend(self.symbolic_executor.run(ctx, specs_to_run))

            # 5. Reconcile: if symbolic check refutes an original diagnostic, suppress/modify it.
            spec_failed: set[str] = set()
            spec_inconclusive: set[str] = set()
            for sd in symbolic_post_diagnostics:
                if not isinstance(sd, dict):
                    continue
                spec_id = sd.get("spec_id")
                if not spec_id and isinstance(sd.get("rule"), str) and "::" in sd.get("rule"):
                    spec_id = sd.get("rule").split("::", 1)[1]
                if not spec_id:
                    continue
                if sd.get("symbolic_result") == "fail":
                    spec_failed.add(str(spec_id))
                elif sd.get("symbolic_result") == "inconclusive":
                    spec_inconclusive.add(str(spec_id))

            reconciled: List[Dict[str, Any]] = []
            for d in diagnostics:
                if not isinstance(d, dict):
                    continue
                spec_ids = d.get("symbolic_cross_checks")
                if not spec_ids:
                    reconciled.append(d)
                    continue
                spec_ids = [str(s) for s in spec_ids]
                any_fail = any(s in spec_failed for s in spec_ids)
                any_inconclusive = any(s in spec_inconclusive for s in spec_ids)

                # Conservative policy:
                # - Keep if any check failed (supports the LLM diagnostic)
                # - Keep if any check is inconclusive
                # - Suppress only if ALL checks pass (no fail, no inconclusive)
                if any_fail:
                    d["symbolic_reconciliation"] = {"status": "supported", "spec_ids": spec_ids}
                    reconciled.append(d)
                elif any_inconclusive:
                    d["symbolic_reconciliation"] = {"status": "inconclusive", "spec_ids": spec_ids}
                    reconciled.append(d)
                else:
                    suppressed_diagnostics.append(
                        {
                            "reason": "symbolic_check_refuted_original",
                            "spec_ids": spec_ids,
                            "original_diagnostic": d,
                        }
                    )

            diagnostics = reconciled
            
        return {
            "id": sample.get("id"),
            "topic": topic.get("name") if topic else None,
            "verifier": verifier_used,
            "diagnostics": diagnostics,
            "symbolic_post_diagnostics": symbolic_post_diagnostics,
            "agentic": {
                "enabled": self.enable_agentic_postcheck,
                "actions": agentic_actions,
                "generated_specs": generated_specs,
                "suppressed_diagnostics": suppressed_diagnostics,
            },
            "score": -1.0 * len(diagnostics) # Simple scoring
        }

    def _agentic_decide_symbolic_check(self, srd: str, diagnostic: Dict[str, Any]) -> Dict[str, Any]:
        """Ask LLM whether a symbolic cross-check should be constructed, and return a safe spec if needed."""
        system_prompt = (
            "You are a cautious physics verification agent. "
            "Your job: decide if a reported rule violation needs an additional symbolic cross-check. "
            "Be conservative: if unsure, say need=false. "
            "You must output ONLY valid JSON."
        )

        available = {
            "power_law": {
                "description": "Check whether dependent ~ independent^p in any equation that contains both variables.",
                "params_schema": {
                    "dependent_candidates": "list[str]",
                    "independent_candidates": "list[str]",
                    "dependent_power": "number (default 1). Interprets check as dependent^dependent_power ~ independent^expected_exponent. Useful for laws written as T^2 ~ r^3.",
                    "expected_exponent": "number",
                    "tolerance": "number (default 0.1)",
                },
            }
            ,
            "multi_power_law": {
                "description": "Check whether dependent^k matches a multi-variable power law product over several independent variables.",
                "params_schema": {
                    "dependent": "str",
                    "independents": "list[str]",
                    "expected_exponents": "dict[str, number] or list[number] aligned with independents",
                    "dependent_power": "number (default 1)",
                    "tolerance": "number (default 0.1)",
                },
            },
            "equation_equivalence": {
                "description": "Check whether any extracted equation is algebraically equivalent to a canonical equation (up to scalar multiple).",
                "params_schema": {
                    "canonical_latex": "list[str] (canonical equations in LaTeX)",
                    "required_symbols": "list[str] (optional; candidate equations must contain all)",
                    "allow_scalar_multiple": "bool (default true)",
                    "allow_additive_constant": "bool (default false). Accepts equivalence up to an additive constant.",
                },
            },
        }

        user_prompt = (
            "Rule SRD:\n" + srd + "\n\n"
            "Reported diagnostic (may be wrong):\n" + json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n\n"
            "Available symbolic primitives:\n" + json.dumps(available, ensure_ascii=False, indent=2) + "\n\n"
            "Return JSON: {\n"
            "  \"need\": true|false,\n"
            "  \"reason\": string,\n"
            "  \"spec\": {\n"
            "     \"spec_id\": string (unique, short),\n"
            "     \"title\": string,\n"
            "     \"description\": string,\n"
            "     \"primitive\": one of ['power_law', 'equation_equivalence'],\n"
            "     \"params\": object\n"
            "  } | null\n"
            "}\n"
        )

        resp = self.rule_verifier._llm_json(system_prompt=system_prompt, user_prompt=user_prompt, fallback={"need": False, "reason": "llm_unavailable", "spec": None})
        if not isinstance(resp, dict):
            return {"need": False, "reason": "invalid_llm_response", "spec": None}
        return resp

    def _record_error_experience(self, sample: Dict, topic: Optional[Dict], errors: List[Dict]):
        experience = {
            "timestamp": datetime.datetime.now().isoformat(),
            "sample_id": sample.get("id"),
            "topic": topic.get("name") if topic else "Unknown",
            "question_snippet": sample.get("question", "")[:200],
            "errors_found": errors
        }
        self.error_experiences.append(experience)
        
        # Append to file immediately
        exp_file = self.results_dir / "error_experiences.json"
        
        all_exps = []
        if exp_file.exists():
            try:
                with open(exp_file, 'r') as f:
                    all_exps = json.load(f)
            except:
                pass
        
        all_exps.append(experience)
        
        with open(exp_file, 'w') as f:
            json.dump(all_exps, f, indent=2, ensure_ascii=False)

    def run_batch(self, samples: List[Dict]):
        results = []
        for s in samples:
            print(f"Verifying sample {s.get('id')}...")
            res = self.verify(s)
            results.append(res)
        return results

if __name__ == "__main__":
    # Test run
    import sys
    
    # Load sample data
    data_path = "data/evaluation_sample_30.json"
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        
    try:
        with open(data_path, 'r') as f:
            samples = json.load(f)
    except FileNotFoundError:
        print(f"File {data_path} not found.")
        sys.exit(1)

    # Limit to first few for testing if needed, or run all
    # samples = samples[:1] 

    verifier = TopDownVerifier()
    results = verifier.run_batch(samples)
    
    # Save results
    with open("results/top_down_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Done. Results saved to results/top_down_results.json")
