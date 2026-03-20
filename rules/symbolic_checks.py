from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import sympy
from sympy.core.relational import Equality, GreaterThan, LessThan, Relational, StrictGreaterThan, StrictLessThan

from rules.base import RuleContext
from symbolic.symbolic_system import EnrichedSymbolGraph, FormulaParser
from symbolic.symbolic_catalog import SymbolicCheckSpec


_TEXT_STOPWORDS = {
    "left",
    "right",
    "mathbf",
    "mathrm",
    "text",
    "cdot",
    "times",
}

def _get_semantic_graph(ctx: RuleContext) -> EnrichedSymbolGraph:
    sem_graph = getattr(ctx, "semantic_graph", None)
    if sem_graph is None:
        sem_graph = EnrichedSymbolGraph(ctx.graph)
        ctx.semantic_graph = sem_graph
    return sem_graph


def _estimate_power(expr: sympy.Expr, var: sympy.Symbol) -> Optional[float]:
    """Estimate exponent p in expr ~ var^p using a ratio test, or return None."""
    try:
        free_syms = expr.free_symbols
        subs_dict = {s: 1 for s in free_syms if s != var}
        simplified = sympy.simplify(expr.subs(subs_dict))
        x = sympy.Symbol("_x_dummy")
        expr_x = simplified.replace(var, x)
        k = sympy.Number(2)
        ratio = sympy.simplify(expr_x.subs(x, k * x) / expr_x)
        if ratio.is_number:
            p = sympy.log(ratio, k)
            if p.is_number and getattr(p, "is_real", False):
                return float(p)
    except Exception:
        return None
    return None


def _normalize_formula_text(text: Any) -> str:
    normalized = str(text or "")
    for token in ["\\left", "\\right", "{", "}", "$", " "]:
        normalized = normalized.replace(token, "")
    return normalized.lower()


def _extract_formula_tokens(text: Any) -> List[str]:
    normalized = _normalize_formula_text(text)
    raw_tokens = []
    for token in re.findall(r"\\[A-Za-z]+|[A-Za-z]+(?:_[A-Za-z0-9]+)?", normalized):
        token = str(token).lstrip("\\")
        if not token or token in _TEXT_STOPWORDS:
            continue
        if token not in raw_tokens:
            raw_tokens.append(token)
    return raw_tokens


def _relation_symbol_from_text(text: Any) -> Optional[str]:
    raw = str(text or "")
    for token in ["<=", ">=", "<", ">", "=", "\\leq", "\\geq", "\\le", "\\ge"]:
        if token in raw:
            return token
    return None


def _normalize_relation_text(text: Any) -> str:
    raw = _normalize_formula_text(text)
    raw = raw.replace("\\leq", "<=").replace("\\le", "<=")
    raw = raw.replace("\\geq", ">=").replace("\\ge", ">=")
    return raw


def _relation_signature(expr: Relational) -> Optional[tuple[bool, sympy.Expr]]:
    try:
        if isinstance(expr, Equality):
            return (False, sympy.simplify(expr.lhs - expr.rhs))
        if isinstance(expr, (StrictLessThan, LessThan)):
            return (isinstance(expr, StrictLessThan), sympy.simplify(expr.lhs - expr.rhs))
        if isinstance(expr, (StrictGreaterThan, GreaterThan)):
            return (isinstance(expr, StrictGreaterThan), sympy.simplify(expr.rhs - expr.lhs))
    except Exception:
        return None
    return None


def _text_contains_required_symbols(text: Any, required_symbols: List[str]) -> bool:
    haystack = _normalize_formula_text(text)
    for symbol in required_symbols:
        needle = _normalize_formula_text(symbol)
        if needle and needle not in haystack:
            return False
    return True


def _match_formula_patterns_in_graph(
    ctx: RuleContext,
    spec: GeneratedSymbolicCheckSpec,
    *,
    patterns: List[Dict[str, Any]],
    required_symbols: List[str],
) -> List[Dict[str, Any]]:
    candidate_ids: List[str] = []
    for fid, node in ctx.graph.formulas.items():
        raw = getattr(node, "raw", None)
        if raw is None:
            continue
        if required_symbols and not _text_contains_required_symbols(raw, required_symbols):
            continue
        candidate_ids.append(fid)
        tokens = _extract_formula_tokens(raw)
        relation = _relation_symbol_from_text(raw)

        for pattern in patterns:
            all_tokens = [_normalize_formula_text(token) for token in (pattern.get("all_tokens") or []) if token]
            pattern_relation = pattern.get("relation")
            if all(token in tokens for token in all_tokens):
                if pattern_relation and relation and pattern_relation not in {relation, relation.replace("\\", "")}:
                    continue
                return [
                    {
                        "severity": "info",
                        "rule": f"agentic_symbolic::{spec.spec_id}",
                        "spec_id": spec.spec_id,
                        "primitive": spec.primitive,
                        "title": spec.title,
                        "symbolic_result": "pass",
                        "symbol": fid,
                        "message": f"Symbolic cross-check passed for '{spec.title}': formula pattern matched.",
                        "evidence": raw,
                    }
                ]

    if candidate_ids:
        fid0 = candidate_ids[0]
        return [
            {
                "severity": "warning",
                "rule": f"agentic_symbolic::{spec.spec_id}",
                "spec_id": spec.spec_id,
                "primitive": spec.primitive,
                "title": spec.title,
                "symbolic_result": "fail",
                "symbol": fid0,
                "message": f"Symbolic cross-check failed for '{spec.title}': candidate formulas did not match the expected pattern.",
                "evidence": ctx.graph.formulas[fid0].raw if fid0 in ctx.graph.formulas else None,
            }
        ]

    return [
        {
            "severity": "info",
            "rule": f"agentic_symbolic::{spec.spec_id}",
            "spec_id": spec.spec_id,
            "primitive": spec.primitive,
            "title": spec.title,
            "symbolic_result": "inconclusive",
            "symbol": None,
            "message": f"Symbolic cross-check inconclusive for '{spec.title}': no parsed formula contains the required symbols {required_symbols}.",
            "evidence": None,
        }
    ]


# ------------------------- Agentic-generated symbolic checks (safe spec + executor) -------------------------


@dataclass
class GeneratedSymbolicCheckSpec:
    """A safe, data-only spec that can be executed by built-in primitives."""

    spec_id: str
    title: str
    description: str
    primitive: str
    params: Dict[str, Any]
    source_rule_id: Optional[str] = None
    source_message_substring: Optional[str] = None


class GeneratedSymbolicCheckRegistry:
    def __init__(self, path: str = "results/agentic_symbolic_checks.json"):
        self.path = Path(path)

    def load(self) -> List[GeneratedSymbolicCheckSpec]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        out: List[GeneratedSymbolicCheckSpec] = []
        for item in data if isinstance(data, list) else []:
            try:
                out.append(GeneratedSymbolicCheckSpec(**item))
            except Exception:
                continue
        return out

    def save(self, specs: List[GeneratedSymbolicCheckSpec]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [s.__dict__ for s in specs]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert(self, spec: GeneratedSymbolicCheckSpec) -> None:
        specs = self.load()
        by_id = {s.spec_id: s for s in specs}
        by_id[spec.spec_id] = spec
        self.save(list(by_id.values()))


class GeneratedSymbolicCheckExecutor:
    """Execute GeneratedSymbolicCheckSpec using built-in primitives (no arbitrary code)."""

    SUPPORTED_PRIMITIVES = {
        "power_law",
        "multi_power_law",
        "equation_equivalence",
        "inequality_consistency",
        "formula_pattern",
    }

    def run(self, ctx: RuleContext, specs: List[GeneratedSymbolicCheckSpec]) -> List[Dict[str, Any]]:
        diags: List[Dict[str, Any]] = []

        sem_graph = _get_semantic_graph(ctx)

        for spec in specs:
            if spec.primitive not in self.SUPPORTED_PRIMITIVES:
                continue

            if spec.primitive == "power_law":
                diags.extend(self._run_power_law(ctx, sem_graph, spec))
            elif spec.primitive == "multi_power_law":
                diags.extend(self._run_multi_power_law(ctx, sem_graph, spec))
            elif spec.primitive == "equation_equivalence":
                diags.extend(self._run_equation_equivalence(ctx, sem_graph, spec))
            elif spec.primitive == "inequality_consistency":
                diags.extend(self._run_inequality_consistency(ctx, sem_graph, spec))
            elif spec.primitive == "formula_pattern":
                diags.extend(self._run_formula_pattern(ctx, sem_graph, spec))

        return diags

    def _run_power_law(self, ctx: RuleContext, sem_graph: EnrichedSymbolGraph, spec: GeneratedSymbolicCheckSpec) -> List[Dict[str, Any]]:
        params = spec.params or {}
        dependent_candidates = params.get("dependent_candidates") or []
        independent_candidates = params.get("independent_candidates") or []
        expected_exponent = params.get("expected_exponent")
        dependent_power = params.get("dependent_power", 1)
        tolerance = float(params.get("tolerance", 0.1))

        if expected_exponent is None:
            return []
        try:
            expected_exponent = float(expected_exponent)
        except Exception:
            return []

        try:
            dependent_power = float(dependent_power)
        except Exception:
            dependent_power = 1.0
        if dependent_power == 0:
            dependent_power = 1.0

        best: Optional[Dict[str, Any]] = None
        any_relationship_found = False
        for dep in dependent_candidates:
            for indep in independent_candidates:
                findings = sem_graph.get_relationship(dep, indep)
                for finding in findings:
                    exponent = finding.get("exponent")
                    fid = finding.get("fid")
                    if exponent is None or fid is None:
                        continue
                    any_relationship_found = True
                    derived = float(exponent) * float(dependent_power)
                    diff = abs(derived - expected_exponent)
                    candidate = {
                        "dep": dep,
                        "indep": indep,
                        "fid": fid,
                        "exponent": float(exponent),
                        "derived": derived,
                        "diff": diff,
                    }
                    if best is None or candidate["diff"] < best["diff"]:
                        best = candidate

        if not any_relationship_found or best is None:
            return [
                {
                    "severity": "info",
                    "rule": f"agentic_symbolic::{spec.spec_id}",
                    "spec_id": spec.spec_id,
                    "primitive": spec.primitive,
                    "title": spec.title,
                    "symbolic_result": "inconclusive",
                    "symbol": None,
                    "message": (
                        f"Symbolic cross-check inconclusive for '{spec.title}': no equation relates the provided variables "
                        f"(dependent={dependent_candidates}, independent={independent_candidates})."
                    ),
                    "evidence": None,
                }
            ]

        fid = best["fid"]
        evidence = ctx.graph.formulas[fid].raw if fid in ctx.graph.formulas else None
        if best["diff"] <= tolerance:
            return [
                {
                    "severity": "info",
                    "rule": f"agentic_symbolic::{spec.spec_id}",
                    "spec_id": spec.spec_id,
                    "primitive": spec.primitive,
                    "title": spec.title,
                    "symbolic_result": "pass",
                    "symbol": fid,
                    "message": (
                        f"Symbolic cross-check passed for '{spec.title}': matched {best['dep']} ~ {best['indep']}^{best['exponent']:.2f} "
                        f"(=> {best['dep']}^{dependent_power:g} ~ {best['indep']}^{best['derived']:.2f})."
                    ),
                    "evidence": evidence,
                    "details": {
                        "dependent": best["dep"],
                        "independent": best["indep"],
                        "expected_exponent": expected_exponent,
                        "derived_exponent": best["derived"],
                        "tolerance": tolerance,
                    },
                }
            ]

        return [
            {
                "severity": "warning",
                "rule": f"agentic_symbolic::{spec.spec_id}",
                "spec_id": spec.spec_id,
                "primitive": spec.primitive,
                "title": spec.title,
                "symbolic_result": "fail",
                "symbol": fid,
                "message": (
                    f"Symbolic cross-check failed for '{spec.title}': expected {best['dep']}^{dependent_power:g} ~ {best['indep']}^{expected_exponent} "
                    f"(±{tolerance}), found {best['dep']} ~ {best['indep']}^{best['exponent']:.2f} (=> {best['dep']}^{dependent_power:g} ~ {best['indep']}^{best['derived']:.2f})."
                ),
                "evidence": evidence,
                "details": {
                    "dependent": best["dep"],
                    "independent": best["indep"],
                    "expected_exponent": expected_exponent,
                    "derived_exponent": best["derived"],
                    "tolerance": tolerance,
                },
            }
        ]

    def _run_multi_power_law(
        self, ctx: RuleContext, sem_graph: EnrichedSymbolGraph, spec: GeneratedSymbolicCheckSpec
    ) -> List[Dict[str, Any]]:
        params = spec.params or {}
        dependent = params.get("dependent")
        independents = params.get("independents") or []
        expected_exponents = params.get("expected_exponents") or {}
        dependent_power = params.get("dependent_power", 1)
        tolerance = float(params.get("tolerance", 0.1))

        if not dependent or not independents:
            return []

        try:
            dependent_power = float(dependent_power)
        except Exception:
            dependent_power = 1.0
        if dependent_power == 0:
            dependent_power = 1.0

        expected_map: Dict[str, float] = {}
        if isinstance(expected_exponents, dict):
            for k, v in expected_exponents.items():
                try:
                    expected_map[str(k)] = float(v)
                except Exception:
                    continue
        elif isinstance(expected_exponents, list):
            for idx, v in enumerate(expected_exponents):
                if idx < len(independents):
                    try:
                        expected_map[str(independents[idx])] = float(v)
                    except Exception:
                        continue

        if not expected_map:
            return []

        best: Optional[Dict[str, Any]] = None
        any_candidate = False
        for fid, expr in sem_graph.parsed_formulas.items():
            syms = {s.name for s in expr.free_symbols}
            if dependent not in syms:
                continue
            if not set(independents).issubset(syms):
                continue

            any_candidate = True
            try:
                dep_sym = sympy.Symbol(dependent)
                solutions = sympy.solve(expr, dep_sym)
                if not solutions:
                    continue
                rhs = solutions[0]
            except Exception:
                continue

            exponents: Dict[str, float] = {}
            complete = True
            for indep in independents:
                p = _estimate_power(rhs, sympy.Symbol(indep))
                if p is None:
                    complete = False
                    break
                exponents[indep] = float(p) * float(dependent_power)

            if not complete:
                continue

            diffs = []
            for indep, expected in expected_map.items():
                found = exponents.get(indep)
                if found is None:
                    diffs.append(float("inf"))
                else:
                    diffs.append(abs(found - expected))
            score = max(diffs) if diffs else float("inf")

            candidate = {
                "fid": fid,
                "exponents": exponents,
                "score": score,
            }
            if best is None or candidate["score"] < best["score"]:
                best = candidate

        if not any_candidate or best is None:
            return [
                {
                    "severity": "info",
                    "rule": f"agentic_symbolic::{spec.spec_id}",
                    "spec_id": spec.spec_id,
                    "primitive": spec.primitive,
                    "title": spec.title,
                    "symbolic_result": "inconclusive",
                    "symbol": None,
                    "message": (
                        f"Symbolic cross-check inconclusive for '{spec.title}': no equation contains dependent={dependent} "
                        f"and independents={independents}."
                    ),
                    "evidence": None,
                }
            ]

        fid = best["fid"]
        evidence = ctx.graph.formulas[fid].raw if fid in ctx.graph.formulas else None
        if best["score"] <= tolerance:
            return [
                {
                    "severity": "info",
                    "rule": f"agentic_symbolic::{spec.spec_id}",
                    "spec_id": spec.spec_id,
                    "primitive": spec.primitive,
                    "title": spec.title,
                    "symbolic_result": "pass",
                    "symbol": fid,
                    "message": f"Symbolic cross-check passed for '{spec.title}': multi-variable power law matched.",
                    "evidence": evidence,
                    "details": {
                        "dependent": dependent,
                        "expected_exponents": expected_map,
                        "derived_exponents": best["exponents"],
                        "tolerance": tolerance,
                    },
                }
            ]

        return [
            {
                "severity": "warning",
                "rule": f"agentic_symbolic::{spec.spec_id}",
                "spec_id": spec.spec_id,
                "primitive": spec.primitive,
                "title": spec.title,
                "symbolic_result": "fail",
                "symbol": fid,
                "message": f"Symbolic cross-check failed for '{spec.title}': multi-variable power law mismatch.",
                "evidence": evidence,
                "details": {
                    "dependent": dependent,
                    "expected_exponents": expected_map,
                    "derived_exponents": best["exponents"],
                    "tolerance": tolerance,
                },
            }
        ]

    def _run_equation_equivalence(
        self, ctx: RuleContext, sem_graph: EnrichedSymbolGraph, spec: GeneratedSymbolicCheckSpec
    ) -> List[Dict[str, Any]]:
        """Check whether any extracted equation is algebraically equivalent to a canonical equation (up to scalar factor)."""

        params = spec.params or {}
        canonical_latex_list = params.get("canonical_latex") or []
        required_symbols = params.get("required_symbols") or []
        allow_scalar_multiple = bool(params.get("allow_scalar_multiple", True))
        allow_additive_constant = bool(params.get("allow_additive_constant", False))

        if not canonical_latex_list:
            return []

        canon_exprs: List[sympy.Expr] = []
        for latex in canonical_latex_list:
            parsed = FormulaParser.parse(str(latex))
            if parsed is None:
                continue
            if isinstance(parsed, sympy.Eq):
                canon_exprs.append(sympy.simplify(parsed.lhs - parsed.rhs))
            else:
                canon_exprs.append(sympy.simplify(parsed))
        if not canon_exprs:
            fallback_patterns = []
            for latex in canonical_latex_list:
                fallback_patterns.append(
                    {
                        "all_tokens": _extract_formula_tokens(latex),
                        "relation": _relation_symbol_from_text(latex),
                    }
                )
            return _match_formula_patterns_in_graph(
                ctx,
                GeneratedSymbolicCheckSpec(
                    spec_id=spec.spec_id,
                    title=spec.title,
                    description=spec.description,
                    primitive="formula_pattern",
                    params=spec.params,
                    source_rule_id=spec.source_rule_id,
                    source_message_substring=spec.source_message_substring,
                ),
                patterns=fallback_patterns,
                required_symbols=[str(s) for s in required_symbols if s],
            )

        # Candidate equations must contain all required symbols if provided.
        candidates: List[tuple[str, sympy.Expr]] = []
        req = {str(s) for s in required_symbols if s}
        for fid, expr in sem_graph.parsed_formulas.items():
            syms = {s.name for s in expr.free_symbols}
            if req and not req.issubset(syms):
                continue
            if isinstance(expr, sympy.Eq):
                candidates.append((fid, sympy.simplify(expr.lhs - expr.rhs)))
            else:
                candidates.append((fid, sympy.simplify(expr)))

        if not candidates:
            return [
                {
                    "severity": "info",
                    "rule": f"agentic_symbolic::{spec.spec_id}",
                    "spec_id": spec.spec_id,
                    "primitive": spec.primitive,
                    "title": spec.title,
                    "symbolic_result": "inconclusive",
                    "symbol": None,
                    "message": (
                        f"Symbolic cross-check inconclusive for '{spec.title}': no parsed equation contains required symbols {sorted(req)}."
                    ),
                    "evidence": None,
                }
            ]

        def equivalent(a: sympy.Expr, b: sympy.Expr) -> bool:
            try:
                a_s = sympy.simplify(a)
                b_s = sympy.simplify(b)
                if sympy.simplify(a_s - b_s) == 0 or sympy.simplify(a_s + b_s) == 0:
                    return True
                if allow_additive_constant:
                    delta = sympy.simplify(a_s - b_s)
                    if len(delta.free_symbols) == 0:
                        return True
                if not allow_scalar_multiple:
                    return False
                if b_s == 0:
                    return False
                ratio = sympy.simplify(a_s / b_s)
                # Equivalent up to scalar multiple if ratio has no free symbols
                return len(ratio.free_symbols) == 0
            except Exception:
                return False

        for fid, cand in candidates:
            for canon in canon_exprs:
                if equivalent(cand, canon):
                    return [
                        {
                            "severity": "info",
                            "rule": f"agentic_symbolic::{spec.spec_id}",
                            "spec_id": spec.spec_id,
                            "primitive": spec.primitive,
                            "title": spec.title,
                            "symbolic_result": "pass",
                            "symbol": fid,
                            "message": f"Symbolic cross-check passed for '{spec.title}': equation equivalence matched.",
                            "evidence": ctx.graph.formulas[fid].raw if fid in ctx.graph.formulas else None,
                        }
                    ]

        # We found relevant equations but none matches the canonical ones.
        fid0, expr0 = candidates[0]
        return [
            {
                "severity": "warning",
                "rule": f"agentic_symbolic::{spec.spec_id}",
                "spec_id": spec.spec_id,
                "primitive": spec.primitive,
                "title": spec.title,
                "symbolic_result": "fail",
                "symbol": fid0,
                "message": f"Symbolic cross-check failed for '{spec.title}': no extracted equation is equivalent to the canonical form.",
                "evidence": ctx.graph.formulas[fid0].raw if fid0 in ctx.graph.formulas else None,
            }
        ]

    def _run_inequality_consistency(
        self, ctx: RuleContext, sem_graph: EnrichedSymbolGraph, spec: GeneratedSymbolicCheckSpec
    ) -> List[Dict[str, Any]]:
        params = spec.params or {}
        canonical_latex_list = params.get("canonical_latex") or []
        required_symbols = [str(s) for s in (params.get("required_symbols") or []) if s]

        canonical_relations: List[Relational] = []
        for latex in canonical_latex_list:
            parsed = FormulaParser.parse(str(latex))
            if isinstance(parsed, Relational):
                canonical_relations.append(parsed)

        if not canonical_relations:
            return [
                {
                    "severity": "info",
                    "rule": f"agentic_symbolic::{spec.spec_id}",
                    "spec_id": spec.spec_id,
                    "primitive": spec.primitive,
                    "title": spec.title,
                    "symbolic_result": "inconclusive",
                    "symbol": None,
                    "message": f"Symbolic cross-check inconclusive for '{spec.title}': canonical inequality could not be parsed.",
                    "evidence": None,
                }
            ]

        candidates: List[tuple[str, Relational]] = []
        for fid, expr in sem_graph.parsed_formulas.items():
            if not isinstance(expr, Relational):
                continue
            raw = ctx.graph.formulas[fid].raw if fid in ctx.graph.formulas else ""
            if required_symbols and not _text_contains_required_symbols(raw, required_symbols):
                continue
            candidates.append((fid, expr))

        if not candidates:
            canonical_texts = [_normalize_relation_text(s) for s in canonical_latex_list]
            for fid, node in ctx.graph.formulas.items():
                raw = getattr(node, "raw", None)
                if raw is None:
                    continue
                norm_raw = _normalize_relation_text(raw)
                if required_symbols and not _text_contains_required_symbols(raw, required_symbols):
                    continue
                if not any(rel in norm_raw for rel in ["<=", ">=", "<", ">"]):
                    continue
                for canon in canonical_texts:
                    if canon and canon in norm_raw:
                        return [
                            {
                                "severity": "info",
                                "rule": f"agentic_symbolic::{spec.spec_id}",
                                "spec_id": spec.spec_id,
                                "primitive": spec.primitive,
                                "title": spec.title,
                                "symbolic_result": "pass",
                                "symbol": fid,
                                "message": f"Symbolic cross-check passed for '{spec.title}': inequality text pattern matched.",
                                "evidence": raw,
                            }
                        ]

            return [
                {
                    "severity": "info",
                    "rule": f"agentic_symbolic::{spec.spec_id}",
                    "spec_id": spec.spec_id,
                    "primitive": spec.primitive,
                    "title": spec.title,
                    "symbolic_result": "inconclusive",
                    "symbol": None,
                    "message": f"Symbolic cross-check inconclusive for '{spec.title}': no inequality with required symbols {required_symbols} was found.",
                    "evidence": None,
                }
            ]

        canonical_signatures = [_relation_signature(rel) for rel in canonical_relations]
        canonical_signatures = [sig for sig in canonical_signatures if sig is not None]

        for fid, rel in candidates:
            rel_sig = _relation_signature(rel)
            if rel_sig is None:
                continue
            for canon_sig in canonical_signatures:
                if canon_sig is None:
                    continue
                same_strict = rel_sig[0] == canon_sig[0]
                same_expr = sympy.simplify(rel_sig[1] - canon_sig[1]) == 0
                if same_strict and same_expr:
                    return [
                        {
                            "severity": "info",
                            "rule": f"agentic_symbolic::{spec.spec_id}",
                            "spec_id": spec.spec_id,
                            "primitive": spec.primitive,
                            "title": spec.title,
                            "symbolic_result": "pass",
                            "symbol": fid,
                            "message": f"Symbolic cross-check passed for '{spec.title}': inequality matched the canonical constraint.",
                            "evidence": ctx.graph.formulas[fid].raw if fid in ctx.graph.formulas else None,
                        }
                    ]

        fid0, _ = candidates[0]
        return [
            {
                "severity": "warning",
                "rule": f"agentic_symbolic::{spec.spec_id}",
                "spec_id": spec.spec_id,
                "primitive": spec.primitive,
                "title": spec.title,
                "symbolic_result": "fail",
                "symbol": fid0,
                "message": f"Symbolic cross-check failed for '{spec.title}': found inequality does not match the canonical constraint.",
                "evidence": ctx.graph.formulas[fid0].raw if fid0 in ctx.graph.formulas else None,
            }
        ]

    def _run_formula_pattern(
        self, ctx: RuleContext, sem_graph: EnrichedSymbolGraph, spec: GeneratedSymbolicCheckSpec
    ) -> List[Dict[str, Any]]:
        params = spec.params or {}
        patterns = [p for p in (params.get("patterns") or []) if isinstance(p, dict)]
        if not patterns:
            patterns = [
                {
                    "all_tokens": _extract_formula_tokens(latex),
                    "relation": _relation_symbol_from_text(latex),
                }
                for latex in (params.get("canonical_latex") or [])
            ]
        required_symbols = [str(s) for s in (params.get("required_symbols") or []) if s]
        return _match_formula_patterns_in_graph(ctx, spec, patterns=patterns, required_symbols=required_symbols)


def catalog_spec_to_generated(spec: SymbolicCheckSpec) -> GeneratedSymbolicCheckSpec:
    return GeneratedSymbolicCheckSpec(
        spec_id=spec.spec_id,
        title=spec.title,
        description=spec.description,
        primitive=spec.primitive,
        params=spec.params,
        source_rule_id=(spec.match_rule_ids[0] if spec.match_rule_ids else None),
        source_message_substring=(spec.match_keywords[0] if spec.match_keywords else None),
    )
