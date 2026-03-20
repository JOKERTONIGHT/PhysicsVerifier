from __future__ import annotations

import re
from hashlib import sha1
from typing import Any, Dict, List, Optional

from symbolic.symbolic_catalog import SymbolicCheckSpec


class RuleSymbolicSpecSynthesizer:
    """Derive conservative symbolic specs directly from top-down rule text.

    This creates a deterministic bottom-up layer before falling back to agentic
    generation. The goal is not perfect coverage, but turning explicit formulas
    and constraints already present in the SRD/check logic into executable specs.
    """

    _FORMULA_RE = re.compile(
        r"((?:[A-Za-z\\][^\n.;]{0,120})"
        r"(?:=|<|>|≤|≥|\\leq|\\geq|\\le|\\ge|∝)"
        r"(?:[^\n.;]{1,120}))"
    )
    _SYMBOL_RE = re.compile(r"\\?[A-Za-z]+(?:_[A-Za-z0-9]+)?(?:\([^()]{1,20}\))?")
    _STOPWORDS = {
        "the", "and", "for", "with", "from", "into", "when", "then", "that", "this",
        "using", "used", "only", "must", "check", "verify", "ensure", "given", "same",
        "under", "where", "while", "problem", "equation", "equations", "formula", "formulas",
        "motion", "force", "energy", "time", "valid", "correct", "incorrect", "constant",
        "horizontal", "vertical", "student", "solution", "apply", "applied", "value",
    }

    def __init__(self) -> None:
        self._topic_cache: Dict[str, Dict[str, List[SymbolicCheckSpec]]] = {}

    def synthesize_topic(self, domain: str, topic: Dict[str, Any]) -> Dict[str, List[SymbolicCheckSpec]]:
        topic_name = str(topic.get("name") or "Unknown")
        cache_key = f"{domain}::{topic_name}"
        cached = self._topic_cache.get(cache_key)
        if cached is not None:
            return cached

        out: Dict[str, List[SymbolicCheckSpec]] = {}
        for rule in topic.get("rules", []) or []:
            rule_id = str(rule.get("id") or "").strip()
            if not rule_id:
                continue
            specs = self._synthesize_rule(domain, topic_name, rule)
            if specs:
                out[rule_id] = specs

        self._topic_cache[cache_key] = out
        return out

    def synthesize_for_rule(self, domain: str, topic: Dict[str, Any], rule_id: str) -> List[SymbolicCheckSpec]:
        if not rule_id:
            return []
        topic_specs = self.synthesize_topic(domain, topic)
        return list(topic_specs.get(rule_id, []))

    def _synthesize_rule(self, domain: str, topic_name: str, rule: Dict[str, Any]) -> List[SymbolicCheckSpec]:
        rule_id = str(rule.get("id") or "").strip()
        title = str(rule.get("title") or "").strip()
        description = str(rule.get("description") or "").strip()
        check_logic = str(rule.get("check_logic") or "").strip()
        text = "\n".join([title, description, check_logic])

        formulas = self._extract_formula_candidates(text)
        specs: List[SymbolicCheckSpec] = []
        seen_ids = set()

        for index, formula in enumerate(formulas[:4]):
            primitive = self._pick_primitive(formula)
            spec = self._build_formula_spec(
                domain=domain,
                topic_name=topic_name,
                rule_id=rule_id,
                title=title,
                formula=formula,
                primitive=primitive,
                ordinal=index,
            )
            if spec and spec.spec_id not in seen_ids:
                specs.append(spec)
                seen_ids.add(spec.spec_id)

        # Common high-value safety rule not usually written as a full equation.
        lowered = text.lower()
        if (
            any(token in lowered for token in ["speed of light", "less than c", "velocity limit"])
            and "v" in self._extract_required_symbols(text)
            and "c" in self._extract_required_symbols(text)
        ):
            spec_id = self._stable_spec_id(rule_id, "ineq", "v<c")
            if spec_id not in seen_ids:
                specs.append(
                    SymbolicCheckSpec(
                        spec_id=spec_id,
                        title=f"{title or rule_id} inequality check",
                        description="Derived from rule text: verify the stated relativistic speed bound.",
                        primitive="inequality_consistency",
                        params={
                            "canonical_latex": ["v < c"],
                            "required_symbols": ["v", "c"],
                        },
                        match_rule_ids=[rule_id],
                        match_keywords=self._keyword_hints(title, description),
                    )
                )

        return specs

    def _extract_formula_candidates(self, text: str) -> List[str]:
        candidates: List[str] = []
        for match in self._FORMULA_RE.finditer(text or ""):
            raw = self._sanitize_formula(match.group(1))
            for formula in self._split_compound_formula(raw):
                if not formula:
                    continue
                if formula not in candidates:
                    candidates.append(formula)
        return candidates

    def _split_compound_formula(self, formula: str) -> List[str]:
        if not formula:
            return []
        text = formula
        connectors = [" and ", " then ", ", ", "; "]
        parts = [text]
        for conn in connectors:
            next_parts: List[str] = []
            for part in parts:
                next_parts.extend([p.strip() for p in part.split(conn) if p.strip()])
            parts = next_parts or parts

        extracted: List[str] = []
        relation_re = re.compile(r"[^,;]{0,120}(?:=|<=|>=|<|>|\\leq|\\geq|\\le|\\ge|∝)[^,;]{1,120}")
        for part in parts:
            found_any = False
            for m in relation_re.finditer(part):
                piece = self._sanitize_formula(m.group(0))
                if piece and piece not in extracted:
                    extracted.append(piece)
                    found_any = True
            if not found_any and re.search(r"=|<=|>=|<|>|\\leq|\\geq|\\le|\\ge|∝", part):
                piece = self._sanitize_formula(part)
                if piece and piece not in extracted:
                    extracted.append(piece)
        return extracted

    def _sanitize_formula(self, formula: str) -> str:
        cleaned = str(formula or "").strip()
        cleaned = cleaned.strip("()[]{} ")
        cleaned = re.sub(r"^(?:e\.g\.|i\.e\.|for example|such as)\s*[:,]?\s*", "", cleaned, flags=re.I)
        cleaned = cleaned.replace("−", "-")
        cleaned = cleaned.replace("≤", "<=").replace("≥", ">=")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned[:160]

    def _pick_primitive(self, formula: str) -> str:
        normalized = formula.lower()
        if any(token in normalized for token in ["<=", ">=", "<", ">", "\\leq", "\\geq"]):
            return "inequality_consistency"
        if any(token in normalized for token in ["\\oint", "\\int", "\\nabla", "\\cdot d", "d\\phi", "d\\phi_b"]):
            return "formula_pattern"
        return "equation_equivalence"

    def _extract_required_symbols(self, text: str) -> List[str]:
        symbols: List[str] = []
        for raw in self._SYMBOL_RE.findall(text or ""):
            token = raw.lstrip("\\")
            low = token.lower()
            if low in self._STOPWORDS:
                continue
            if len(token) == 1 or any(ch.isupper() for ch in token) or "_" in token or "(" in token:
                if token not in symbols:
                    symbols.append(token)
        return symbols[:8]

    def _keyword_hints(self, title: str, description: str) -> List[str]:
        text = f"{title} {description}".strip()
        parts = re.findall(r"[A-Za-z][A-Za-z'/-]{2,}", text)
        hints: List[str] = []
        for part in parts:
            low = part.lower()
            if low in self._STOPWORDS:
                continue
            if part not in hints:
                hints.append(part)
            if len(hints) >= 4:
                break
        return hints

    def _build_formula_spec(
        self,
        *,
        domain: str,
        topic_name: str,
        rule_id: str,
        title: str,
        formula: str,
        primitive: str,
        ordinal: int,
    ) -> Optional[SymbolicCheckSpec]:
        required_symbols = self._extract_required_symbols(formula)
        if len(required_symbols) < 2:
            return None

        spec_id = self._stable_spec_id(rule_id, primitive, formula, ordinal)
        description = f"Derived from top-down rule text in {domain}/{topic_name}: {formula}"
        keyword_hints = self._keyword_hints(title, formula)

        if primitive == "equation_equivalence":
            params: Dict[str, Any] = {
                "canonical_latex": [formula],
                "required_symbols": required_symbols,
                "allow_scalar_multiple": False,
                "allow_additive_constant": False,
            }
        elif primitive == "inequality_consistency":
            params = {
                "canonical_latex": [formula],
                "required_symbols": required_symbols,
            }
        else:
            params = {
                "patterns": [
                    {
                        "all_tokens": self._extract_required_symbols(formula),
                        "relation": "=" if "=" in formula else None,
                    }
                ],
                "required_symbols": required_symbols,
            }

        return SymbolicCheckSpec(
            spec_id=spec_id,
            title=f"Derived symbolic check for {title or rule_id}",
            description=description,
            primitive=primitive,
            params=params,
            match_rule_ids=[rule_id],
            match_keywords=keyword_hints,
        )

    def _stable_spec_id(self, rule_id: str, primitive: str, formula: str, ordinal: int = 0) -> str:
        digest = sha1(f"{rule_id}|{primitive}|{ordinal}|{formula}".encode("utf-8")).hexdigest()[:10]
        return f"derived_{rule_id}_{digest}"