import json
import datetime
import re
import math
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from core.unified_semantic_matcher import UnifiedSemanticMatcher
from core.unified_retrieval import norm_text
from core.rule_based_verifier import RuleBasedVerifier, RuleContext
from rules.symbolic_checks import (
    GeneratedSymbolicCheckExecutor,
    GeneratedSymbolicCheckRegistry,
    GeneratedSymbolicCheckSpec,
    catalog_spec_to_generated,
)
from symbolic.symbolic_catalog import SymbolicCatalog, SymbolicCheckSpec
from symbolic.experience_bank import SymbolicExperienceBank
from symbolic.spec_synthesis import RuleSymbolicSpecSynthesizer

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

        # Unified catalog takes priority when available
        self._unified_mode = False
        if unified_rules_path and Path(unified_rules_path).exists():
            self.rules_catalog_path = unified_rules_path
            self._unified_mode = True
        else:
            self.rules_catalog_path = rules_catalog_path

        self.catalog = self._load_catalog()
        meta = self.catalog.get("metadata") if isinstance(self.catalog, dict) else {}
        self._unified_v2_mode = bool(
            self._unified_mode and isinstance(meta, dict) and meta.get("catalog_type") == "unified_rules_v2"
        )
        self.topics = self._flatten_topics()
        self.semantic_matcher = (
            UnifiedSemanticMatcher(model=str(self.llm_model or ""))
            if self._unified_v2_mode
            else None
        )
        
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
        # In unified mode, experience rules are already part of the catalog;
        # the separate experience pipeline is only needed when NOT using unified catalog.
        self.enable_experience_pipeline = bool(enable_experience_pipeline) and not self._unified_mode
        self.experience_rules_path = str(experience_rules_path)
        # Keep compatibility with run_top_down.py arguments.
        self.experience_code_manifest_path = str(experience_code_manifest_path)
        self.experience_code_module = str(experience_code_module)
        # Backward compatible registry (results/*) is kept for audit logs, but the source of truth is catalogs/symbolic_catalog.json
        self.symbolic_registry = GeneratedSymbolicCheckRegistry(path=str(self.results_dir / "agentic_symbolic_checks.json"))
        self.symbolic_catalog = SymbolicCatalog(path="catalogs/symbolic_catalog.json")
        self.symbolic_executor = GeneratedSymbolicCheckExecutor()
        self.symbolic_experience_bank = SymbolicExperienceBank(path=str(self.results_dir / "rule_experience_bank.json"))
        self.spec_synthesizer = RuleSymbolicSpecSynthesizer()
        self.experience_rules_index = self._load_experience_rules(self.experience_rules_path) if self.enable_experience_pipeline else {}

        self.error_experiences = []

    def _normalize_topic_key(self, domain: str, topic: str) -> str:
        d = str(domain or "Unknown").strip().lower()
        t_raw = str(topic or "Unknown").strip()
        # Distilled experience topic may look like "Domain / Topic"; keep the last segment for robust matching.
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

    def _extract_keywords(self, text: str, max_kw: int = 8) -> List[str]:
        tokens = re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]+", str(text or ""))
        out: List[str] = []
        for tok in tokens:
            tok = tok.strip()
            if len(tok) < 2:
                continue
            if tok.lower() in {"check", "logic", "rule", "error", "with", "from", "this"}:
                continue
            if tok not in out:
                out.append(tok)
            if len(out) >= max_kw:
                break
        return out

    def _experience_rule_triggered(self, rule: Dict[str, Any], text_all: str) -> Dict[str, Any]:
        hay = str(text_all or "").lower()
        trigger_text = " ".join([
            str(rule.get("title") or ""),
            str(rule.get("trigger") or ""),
            str(rule.get("check_logic") or ""),
        ])
        keywords = self._extract_keywords(trigger_text, max_kw=10)
        hits = [kw for kw in keywords if kw.lower() in hay]
        # Conservative trigger: at least two keyword hits, or one long keyword.
        if len(hits) >= 2:
            return {"triggered": True, "hits": hits}
        if len(hits) == 1 and len(hits[0]) >= 6:
            return {"triggered": True, "hits": hits}
        return {"triggered": False, "hits": hits}

    def _build_experience_symbolic_spec(self, rule: Dict[str, Any]) -> Optional[GeneratedSymbolicCheckSpec]:
        hint = rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {}
        primitive = str(hint.get("primitive") or "none").strip()
        if primitive in {"", "none"}:
            return None

        canonical = str(hint.get("canonical") or "").strip()
        required_symbols = [str(s) for s in (hint.get("required_symbols") or []) if str(s).strip()]

        params: Dict[str, Any] = {}
        if primitive in {"equation_equivalence", "inequality_consistency"}:
            if not canonical or len(required_symbols) < 2:
                return None
            params = {
                "canonical_latex": [canonical],
                "required_symbols": required_symbols,
                "allow_scalar_multiple": False,
                "allow_additive_constant": False,
            }
        elif primitive == "formula_pattern":
            if len(required_symbols) < 2:
                return None
            relation = "=" if "=" in canonical else None
            params = {
                "patterns": [{"all_tokens": required_symbols, "relation": relation}],
                "required_symbols": required_symbols,
            }
        else:
            # Unsupported primitive in experience_hint for this pipeline version.
            return None

        rule_id = str(rule.get("rule_id") or "exp_unknown")
        return GeneratedSymbolicCheckSpec(
            spec_id=f"exp_sym_{rule_id}",
            title=f"Experience symbolic check: {rule.get('title')}",
            description=str(rule.get("check_logic") or ""),
            primitive=primitive,
            params=params,
            source_rule_id=rule_id,
            source_message_substring=str(rule.get("title") or ""),
        )

    def _build_rule_context(self, sample: Dict[str, Any]) -> RuleContext:
        text_all = "\n".join([
            sample.get("question", ""),
            sample.get("context", ""),
            sample.get("prediction", ""),
            sample.get("answer", ""),
        ])
        parsed = self.rule_verifier._extract_symbols_and_formulas(text_all)
        graph = self.rule_verifier._build_symbol_graph(parsed["lines"], parsed["symbols"], parsed["formulas"])
        return RuleContext(
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

    def _prepare_unified_v2_rule(self, rule: Dict[str, Any]) -> Dict[str, Any]:
        prepared = dict(rule)
        rid = str(rule.get("rule_id") or rule.get("id") or "").strip()
        prepared["id"] = rid
        if not prepared.get("description"):
            parts = []
            if rule.get("trigger"):
                parts.append(f"Trigger: {rule.get('trigger')}")
            if rule.get("check_logic"):
                parts.append(f"Check Logic: {rule.get('check_logic')}")
            prepared["description"] = "\n".join(parts)
        return prepared

    def _build_unified_v2_topic_for_synthesis(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        adapted = dict(topic)
        adapted["rules"] = [self._prepare_unified_v2_rule(rule) for rule in (topic.get("rules") or []) if isinstance(rule, dict)]
        return adapted

    # ---- Unified-mode SRD helpers ----

    @staticmethod
    def _build_srd_for_rule(r: Dict[str, Any]) -> str:
        """Construct an SRD prompt string from a rule dict, adapting to its source type."""
        source = r.get("source", "knowledge")
        if source == "experience_tagged":
            # Tagged experience rules have very detailed descriptions that serve as full SRDs
            return r.get("description", "")
        elif source == "experience":
            # Distilled experience: use trigger + check_logic
            parts = []
            if r.get("title"):
                parts.append(f"Title: {r['title']}")
            if r.get("trigger"):
                parts.append(f"Trigger: {r['trigger']}")
            if r.get("check_logic"):
                parts.append(f"Check Logic: {r['check_logic']}")
            return "\n".join(parts)
        elif r.get("rule_id") or (r.get("id") and not r.get("source")):
            parts = []
            if r.get("title"):
                parts.append(f"Title: {r['title']}")
            if r.get("trigger"):
                parts.append(f"Trigger: {r['trigger']}")
            if r.get("check_logic"):
                parts.append(f"Check Logic: {r['check_logic']}")
            if r.get("description") and not parts:
                parts.append(str(r.get("description") or ""))
            return "\n".join(parts)
        else:
            # Knowledge rules: classic Title + Description + Check Logic
            return f"Title: {r.get('title')}\nDescription: {r.get('description')}\nCheck Logic: {r.get('check_logic')}"

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

        topic: Optional[Dict[str, Any]] = None
        diagnostics: List[Dict[str, Any]] = []
        used_rules: List[str] = []
        verifier_used = "top_down_rule_based"
        agentic_actions: List[Dict[str, Any]] = []
        symbolic_post_diagnostics: List[Dict[str, Any]] = []
        generated_specs: List[Dict[str, Any]] = []
        suppressed_diagnostics: List[Dict[str, Any]] = []
        experience_candidates: Dict[int, List[SymbolicCheckSpec]] = {}
        topic_synthesized_specs: Dict[str, List[SymbolicCheckSpec]] = {}
        experience_post_diagnostics: List[Dict[str, Any]] = []
        experience_symbolic_post_diagnostics: List[Dict[str, Any]] = []
        experience_actions: List[Dict[str, Any]] = []

        selected_rule_records: List[Dict[str, Any]] = []
        retrieved_topics_payload: List[Dict[str, Any]] = []
        retrieved_clusters_payload: List[Dict[str, Any]] = []
        retrieved_rules_payload: List[Dict[str, Any]] = []
        semantic_selection_error = ""
        selection_strategy = "semantic_unavailable"
        topic_matches: List[Dict[str, Any]] = []

        if self._unified_v2_mode:
            semantic_result: Optional[Dict[str, Any]] = None
            if self.semantic_matcher is not None and self.semantic_matcher.available:
                try:
                    semantic_result = self.semantic_matcher.select_tree_semantically(sample, self.catalog)
                    selection_strategy = "semantic_tree_selection"
                except Exception as exc:
                    semantic_selection_error = f"{type(exc).__name__}: {exc}"
            else:
                semantic_selection_error = "Semantic matcher is not available."

            if semantic_result is not None:
                verifier_used = "unified_v2_semantic_rule_based"
                topic_matches = list(semantic_result.get("selected_topics") or [])
                semantic_topic_index = {
                    (str(item.get("domain") or "Unknown"), str(item.get("topic") or "Unknown")): item
                    for item in topic_matches
                    if isinstance(item, dict)
                }
                retrieved_topics_payload = [
                    {
                        "domain": str(item.get("domain") or "Unknown"),
                        "topic": str(item.get("topic") or "Unknown"),
                        "score": float(item.get("score") or 0.0),
                        "evidence": {"reason": str(item.get("reason") or "")},
                    }
                    for item in topic_matches
                ]
                retrieved_clusters_payload = [
                    {
                        "domain": str(item.get("domain") or "Unknown"),
                        "topic": str(item.get("topic") or "Unknown"),
                        "cluster_id": str(item.get("cluster_id") or ""),
                        "cluster": str(item.get("cluster") or ""),
                        "score": float(item.get("score") or 0.0),
                        "reason": str(item.get("reason") or ""),
                    }
                    for item in (semantic_result.get("selected_clusters") or [])
                    if isinstance(item, dict)
                ]
                if topic_matches:
                    raw_topic = topic_matches[0].get("topic_obj") if isinstance(topic_matches[0].get("topic_obj"), dict) else None
                    topic = dict(raw_topic) if isinstance(raw_topic, dict) else None
                    if topic is not None and not topic.get("domain"):
                        topic["domain"] = str(topic_matches[0].get("domain") or "Unknown")
                    if topic:
                        print(f"Retrieved primary topic: {topic['domain']} - {topic['name']}")
                selected_rule_records = []
                for item in (semantic_result.get("selected_rules") or []):
                    if not isinstance(item, dict):
                        continue
                    raw_rule = item.get("rule_obj") if isinstance(item.get("rule_obj"), dict) else None
                    if not raw_rule:
                        continue
                    rule = self._prepare_unified_v2_rule(raw_rule)
                    topic_obj = None
                    topic_key = (str(item.get("domain") or "Unknown"), str(item.get("topic") or "Unknown"))
                    topic_match = semantic_topic_index.get(topic_key)
                    raw_topic_obj = topic_match.get("topic_obj") if isinstance(topic_match, dict) and isinstance(topic_match.get("topic_obj"), dict) else None
                    if isinstance(raw_topic_obj, dict):
                        topic_obj = dict(raw_topic_obj)
                        if not topic_obj.get("domain"):
                            topic_obj["domain"] = topic_key[0]
                    selected_rule_records.append(
                        {
                            "domain": str(item.get("domain") or "Unknown"),
                            "topic_name": str(item.get("topic") or "Unknown"),
                            "topic": topic_obj,
                            "cluster_id": str(item.get("cluster_id") or ""),
                            "cluster": str(item.get("cluster") or ""),
                            "rule": rule,
                            "score": float(item.get("score") or 0.0),
                            "scope": str(rule.get("scope") or "domain"),
                            "evidence": {"reason": str(item.get("reason") or "")},
                            "manual_override_reason": "",
                        }
                    )
                retrieved_rules_payload = [
                    {
                        "rule_id": str(item["rule"].get("id") or ""),
                        "domain": str(item.get("domain") or "Unknown"),
                        "topic": str(item.get("topic_name") or "Unknown"),
                        "cluster_id": str(item.get("cluster_id") or ""),
                        "cluster": str(item.get("cluster") or ""),
                        "title": str(item["rule"].get("title") or ""),
                        "scope": str(item.get("scope") or item["rule"].get("scope") or "domain"),
                        "score": float(item.get("score") or 0.0),
                        "manual_override_reason": "",
                        "evidence": item.get("evidence") or {},
                    }
                    for item in selected_rule_records
                ]

            current_translations: Dict[str, Dict[str, str]] = {}
            rule_ids: List[str] = []
            for item in selected_rule_records:
                rule = item["rule"]
                rid = str(rule.get("id") or "").strip()
                if not rid:
                    continue
                rule_ids.append(rid)
                current_translations[rid] = {"srd": self._build_srd_for_rule(rule)}

            topic_synthesized_specs = {}
            seen_topic_keys = set()
            for item in topic_matches:
                topic_obj = None
                if isinstance(item.get("topic_obj"), dict):
                    topic_obj = dict(item.get("topic_obj"))
                    if not topic_obj.get("domain"):
                        topic_obj["domain"] = str(item.get("domain") or "Unknown")
                elif isinstance(item.get("topic"), dict):
                    topic_obj = item.get("topic")
                if not topic_obj:
                    continue
                topic_key = self._normalize_topic_key(topic_obj.get("domain"), topic_obj.get("name"))
                if topic_key in seen_topic_keys:
                    continue
                seen_topic_keys.add(topic_key)
                adapted_topic = self._build_unified_v2_topic_for_synthesis(topic_obj)
                topic_synthesized_specs.update(
                    self.spec_synthesizer.synthesize_topic(topic_obj.get("domain", "Unknown"), adapted_topic)
                )

            if rule_ids:
                self.rule_verifier.rules_to_check = rule_ids
                self.rule_verifier.rule_translations = current_translations
                print(f"Running unified v2 rule check with {len(rule_ids)} rules...")
                result = self.rule_verifier.analyze(sample)
                diagnostics = result.get("diagnostics", [])
                used_rules = rule_ids
            else:
                self.rule_verifier.rules_to_check = []
                self.rule_verifier.rule_translations = {}
        else:
            # 1. Classify
            topic = self.classify_topic(question)

            if topic:
                print(f"Classified into: {topic['domain']} - {topic['name']}")
                rules = topic.get("rules", [])
                topic_synthesized_specs = self.spec_synthesizer.synthesize_topic(topic.get("domain", "Unknown"), topic)

                if rules:
                    # 2. Prepare Rule Verifier
                    # Convert catalog rules to the format expected by RuleBasedVerifier
                    # We use source-aware SRD construction
                    current_translations = {}
                    rule_ids = []
                    for r in rules:
                        rid = r.get("id")
                        if not rid:
                            continue
                        rule_ids.append(rid)
                        current_translations[rid] = {
                            "srd": self._build_srd_for_rule(r)
                        }

                    self.rule_verifier.rules_to_check = rule_ids
                    self.rule_verifier.rule_translations = current_translations

                    # 3. Run Rule Check
                    print(f"Running rule check with {len(rule_ids)} rules...")
                    result = self.rule_verifier.analyze(sample)
                    diagnostics = result.get("diagnostics", [])
                    used_rules = rule_ids
                    verifier_used = "top_down_rule_based" if not self._unified_mode else "unified_rule_based"
            else:
                print("Could not classify topic or no topic found.")

        ctx: Optional[RuleContext] = None
        if (self.enable_agentic_postcheck and diagnostics) or (self.enable_experience_pipeline and topic):
            ctx = self._build_rule_context(sample)

        # 4. Agentic post-check: for each diagnostic, decide whether to build a symbolic cross-check spec.
        # Build a lookup from rule id -> rule dict for symbolic_hint access (unified mode)
        _rule_by_id: Dict[str, Dict[str, Any]] = {}
        _rule_topic_by_id: Dict[str, Dict[str, Any]] = {}
        if self._unified_v2_mode:
            for item in selected_rule_records:
                rule = item.get("rule") if isinstance(item.get("rule"), dict) else None
                if not rule:
                    continue
                rid = rule.get("id")
                if not rid:
                    continue
                _rule_by_id[str(rid)] = rule
                _rule_topic_by_id[str(rid)] = {
                    "domain": str(item.get("domain") or "Unknown"),
                    "name": str(item.get("topic_name") or "Unknown"),
                }
        elif topic:
            for _r in (topic.get("rules") or []):
                _rid = _r.get("id")
                if _rid:
                    _rule_by_id[_rid] = _r
                    _rule_topic_by_id[_rid] = {
                        "domain": str(topic.get("domain") or "Unknown"),
                        "name": str(topic.get("name") or "Unknown"),
                    }

        if self.enable_agentic_postcheck and diagnostics and ctx is not None:

            checks_used = 0
            for d in diagnostics:
                if not isinstance(d, dict):
                    continue
                rid = d.get("rule")
                if not rid:
                    continue

                # 4.1 Try to find an existing symbolic check from catalog first.
                matched_topic = _rule_topic_by_id.get(str(rid), topic or {"domain": "Unknown", "name": "Unknown"})
                domain_name = matched_topic.get("domain") if isinstance(matched_topic, dict) else "Unknown"
                topic_name = matched_topic.get("name") if isinstance(matched_topic, dict) else "Unknown"
                existing = self.symbolic_catalog.find_applicable(domain=domain_name, topic=topic_name, diagnostic=d)
                promoted = self.symbolic_experience_bank.get_promoted_specs(domain=domain_name, topic=topic_name, rule_id=rid)
                synthesized = list(topic_synthesized_specs.get(rid, []))

                # In unified mode, experience rules may carry symbolic_hint –
                # convert it into a GeneratedSymbolicCheckSpec and include it.
                inline_hint_specs: List[GeneratedSymbolicCheckSpec] = []
                rule_obj = _rule_by_id.get(rid)
                if rule_obj and rule_obj.get("symbolic_hint"):
                    hint_spec = self._build_experience_symbolic_spec_from_hint(
                        rule_id=rid,
                        title=rule_obj.get("title", ""),
                        check_logic=rule_obj.get("check_logic", ""),
                        symbolic_hint=rule_obj["symbolic_hint"],
                    )
                    if hint_spec is not None:
                        inline_hint_specs.append(hint_spec)

                resolved_specs = self._merge_specs(
                    [catalog_spec_to_generated(s) for s in existing],
                    [catalog_spec_to_generated(s) for s in promoted],
                    [catalog_spec_to_generated(s) for s in synthesized],
                    inline_hint_specs,
                )

                if synthesized:
                    experience_candidates[id(d)] = synthesized

                if resolved_specs:
                    spec_ids = [s.spec_id for s in resolved_specs]
                    d.setdefault("symbolic_cross_checks", []).extend(spec_ids)
                    run_result = self.symbolic_executor.run(ctx, resolved_specs)
                    symbolic_post_diagnostics.extend(run_result)
                    agentic_actions.append(
                        {
                            "need": False,
                            "reason": "used_resolved_symbolic_checks",
                            "diagnostic_rule": rid,
                            "spec_ids": spec_ids,
                            "sources": {
                                "catalog": [s.spec_id for s in existing],
                                "experience_bank": [s.spec_id for s in promoted],
                                "derived_from_rule": [s.spec_id for s in synthesized],
                                "inline_hint": [s.spec_id for s in inline_hint_specs],
                            },
                        }
                    )

                    if any(r.get("symbolic_result") in {"pass", "fail"} for r in run_result if isinstance(r, dict)):
                        continue

                if checks_used >= self.agentic_max_checks_per_sample or not self.rule_verifier._llm_available():
                    continue

                srd = None
                rule_obj = _rule_by_id.get(str(rid))
                if rule_obj:
                    srd = self._build_srd_for_rule(rule_obj)
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

                self.symbolic_registry.upsert(spec)
                generated_specs.append(spec.__dict__)
                d.setdefault("symbolic_cross_checks", []).append(spec.spec_id)
                checks_used += 1
                experience_candidates.setdefault(id(d), []).append(
                    SymbolicCheckSpec(
                        spec_id=spec.spec_id,
                        title=spec.title,
                        description=spec.description,
                        primitive=spec.primitive,
                        params=spec.params,
                        match_rule_ids=[rid],
                        match_keywords=[str(d.get("message") or "")[:60]],
                    )
                )

            # Execute newly generated specs (avoid re-running the entire registry every time)
            if generated_specs:
                specs_to_run = [GeneratedSymbolicCheckSpec(**s) for s in generated_specs]
                symbolic_post_diagnostics.extend(self.symbolic_executor.run(ctx, specs_to_run))

            symbolic_post_diagnostics = self._dedupe_symbolic_post_diagnostics(symbolic_post_diagnostics)

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
                    self._record_symbolic_experience(
                        sample=sample,
                        topic=_rule_topic_by_id.get(str(d.get("rule")), topic),
                        diagnostic=d,
                        outcome="no_symbolic_match",
                        proposed_specs=experience_candidates.get(id(d), []),
                    )
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
                    self._record_symbolic_experience(
                        sample=sample,
                        topic=_rule_topic_by_id.get(str(d.get("rule")), topic),
                        diagnostic=d,
                        outcome="supported",
                        proposed_specs=experience_candidates.get(id(d), []),
                    )
                elif any_inconclusive:
                    d["symbolic_reconciliation"] = {"status": "inconclusive", "spec_ids": spec_ids}
                    reconciled.append(d)
                    self._record_symbolic_experience(
                        sample=sample,
                        topic=_rule_topic_by_id.get(str(d.get("rule")), topic),
                        diagnostic=d,
                        outcome="inconclusive",
                        proposed_specs=experience_candidates.get(id(d), []),
                    )
                else:
                    suppressed_diagnostics.append(
                        {
                            "reason": "symbolic_check_refuted_original",
                            "spec_ids": spec_ids,
                            "original_diagnostic": d,
                        }
                    )
                    self._record_symbolic_experience(
                        sample=sample,
                        topic=_rule_topic_by_id.get(str(d.get("rule")), topic),
                        diagnostic=d,
                        outcome="suppressed",
                        proposed_specs=experience_candidates.get(id(d), []),
                    )

            diagnostics = reconciled

        # 6. Experience-rule pipeline: add bottom-up experience diagnostics and corresponding symbolic checks.
        if self.enable_experience_pipeline and topic and ctx is not None:
            exp_rules = self._get_experience_rules_for_topic(topic)
            if exp_rules:
                text_all = "\n".join([
                    sample.get("question", ""),
                    sample.get("context", ""),
                    sample.get("prediction", ""),
                    sample.get("answer", ""),
                ])
                # Keep per-sample overhead bounded.
                for rule in exp_rules[:40]:
                    trig = self._experience_rule_triggered(rule, text_all)
                    if not trig.get("triggered"):
                        continue

                    rule_id = str(rule.get("rule_id") or "exp_unknown")
                    diag = {
                        "severity": "warning",
                        "rule": f"experience::{rule_id}",
                        "symbol": None,
                        "message": f"Experience rule matched: {rule.get('title')}",
                        "evidence": ",".join(trig.get("hits") or [])[:200],
                        "experience_rule": {
                            "title": rule.get("title"),
                            "trigger": rule.get("trigger"),
                            "check_logic": rule.get("check_logic"),
                            "error_type": rule.get("error_type"),
                        },
                    }

                    spec = self._build_experience_symbolic_spec(rule)
                    if spec is not None:
                        diag["experience_symbolic_cross_checks"] = [spec.spec_id]
                        run_result = self.symbolic_executor.run(ctx, [spec])
                        experience_symbolic_post_diagnostics.extend(run_result)

                        result_flag = None
                        for rr in run_result:
                            if isinstance(rr, dict) and rr.get("spec_id") == spec.spec_id:
                                result_flag = rr.get("symbolic_result")
                                break

                        if result_flag == "fail":
                            # Experience symbolic check disagrees with the triggered experience rule -> suppress.
                            experience_actions.append(
                                {
                                    "rule_id": rule_id,
                                    "action": "suppressed_by_symbolic",
                                    "spec_id": spec.spec_id,
                                }
                            )
                            continue
                        if result_flag in {"pass", "inconclusive"}:
                            diag["experience_symbolic_reconciliation"] = {
                                "status": "supported" if result_flag == "pass" else "inconclusive",
                                "spec_ids": [spec.spec_id],
                            }

                    experience_post_diagnostics.append(diag)

                # Merge experience diagnostics into final diagnostics.
                diagnostics.extend(experience_post_diagnostics)
            
        return {
            "id": sample.get("id"),
            "topic": topic.get("name") if topic else None,
            "verifier": verifier_used,
            "unified_mode": self._unified_mode,
            "selection_strategy": selection_strategy,
            "semantic_selection_error": semantic_selection_error,
            "retrieved_topics": retrieved_topics_payload,
            "retrieved_clusters": retrieved_clusters_payload,
            "retrieved_rules": retrieved_rules_payload,
            "diagnostics": diagnostics,
            "symbolic_post_diagnostics": symbolic_post_diagnostics,
            "experience_post_diagnostics": experience_post_diagnostics,
            "experience_symbolic_post_diagnostics": experience_symbolic_post_diagnostics,
            "agentic": {
                "enabled": self.enable_agentic_postcheck,
                "actions": agentic_actions,
                "generated_specs": generated_specs,
                "suppressed_diagnostics": suppressed_diagnostics,
            },
            "experience_pipeline": {
                "enabled": self.enable_experience_pipeline,
                "actions": experience_actions,
                "rules_path": self.experience_rules_path,
            },
            "score": -1.0 * len(diagnostics) # Simple scoring
        }

    def _merge_specs(self, *groups: List[GeneratedSymbolicCheckSpec]) -> List[GeneratedSymbolicCheckSpec]:
        merged: List[GeneratedSymbolicCheckSpec] = []
        seen = set()
        for group in groups:
            for spec in group or []:
                spec_id = getattr(spec, "spec_id", None)
                if not spec_id or spec_id in seen:
                    continue
                merged.append(spec)
                seen.add(spec_id)
        return merged

    @staticmethod
    def _build_experience_symbolic_spec_from_hint(
        *,
        rule_id: str,
        title: str,
        check_logic: str,
        symbolic_hint: Dict[str, Any],
    ) -> Optional[GeneratedSymbolicCheckSpec]:
        """Convert an inline symbolic_hint from a unified rule into a GeneratedSymbolicCheckSpec."""
        primitive = str(symbolic_hint.get("primitive") or "none").strip()
        if primitive in {"", "none"}:
            return None

        canonical = str(symbolic_hint.get("canonical") or "").strip()
        required_symbols = [str(s) for s in (symbolic_hint.get("required_symbols") or []) if str(s).strip()]

        params: Dict[str, Any] = {}
        if primitive in {"equation_equivalence", "inequality_consistency"}:
            if not canonical or len(required_symbols) < 2:
                return None
            params = {
                "canonical_latex": [canonical],
                "required_symbols": required_symbols,
                "allow_scalar_multiple": False,
                "allow_additive_constant": False,
            }
        elif primitive == "formula_pattern":
            if len(required_symbols) < 2:
                return None
            relation = "=" if "=" in canonical else None
            params = {
                "patterns": [{"all_tokens": required_symbols, "relation": relation}],
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

        return GeneratedSymbolicCheckSpec(
            spec_id=f"unified_hint_{rule_id}",
            title=f"Symbolic hint: {title}",
            description=check_logic,
            primitive=primitive,
            params=params,
            source_rule_id=rule_id,
            source_message_substring=title[:200] if title else "",
        )

    def _dedupe_symbolic_post_diagnostics(self, diagnostics: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        unique: List[Dict[str, Any]] = []
        seen = set()
        for item in diagnostics or []:
            if not isinstance(item, dict):
                continue
            key = (
                item.get("spec_id"),
                item.get("symbolic_result"),
                item.get("symbol"),
                item.get("message"),
                item.get("evidence"),
            )
            if key in seen:
                continue
            unique.append(item)
            seen.add(key)
        return unique

    def _record_symbolic_experience(
        self,
        *,
        sample: Dict[str, Any],
        topic: Optional[Dict[str, Any]],
        diagnostic: Dict[str, Any],
        outcome: str,
        proposed_specs: List[SymbolicCheckSpec],
    ) -> None:
        if not topic or not isinstance(diagnostic, dict):
            return
        spec_ids = [str(s) for s in (diagnostic.get("symbolic_cross_checks") or []) if s]
        self.symbolic_experience_bank.record_event(
            domain=str(topic.get("domain") or "Unknown"),
            topic=str(topic.get("name") or "Unknown"),
            rule_id=str(diagnostic.get("rule") or "<unknown>"),
            diagnostic=diagnostic,
            outcome=outcome,
            had_symbolic_match=bool(spec_ids),
            spec_ids=spec_ids,
            proposed_specs=proposed_specs,
        )

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
            "inequality_consistency": {
                "description": "Check whether an extracted inequality is equivalent to a canonical safety or validity constraint.",
                "params_schema": {
                    "canonical_latex": "list[str] (canonical inequalities in LaTeX or plain text)",
                    "required_symbols": "list[str] (candidate inequalities must contain all)",
                },
            },
            "formula_pattern": {
                "description": "Conservative text-pattern matcher for formulas that are hard to parse symbolically, such as vector/integral forms.",
                "params_schema": {
                    "patterns": "list[object] with all_tokens and optional relation",
                    "required_symbols": "list[str]",
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
            "     \"primitive\": one of ['power_law', 'multi_power_law', 'equation_equivalence', 'inequality_consistency', 'formula_pattern'],\n"
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
