from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import sympy

from rules.base import RulePlugin, RuleContext, RuleRuntime
from symbolic.symbolic_system import EnrichedSymbolGraph, FormulaParser
from symbolic.symbolic_catalog import SymbolicCheckSpec


# ------------------------- Manual symbolic checks (existing) -------------------------


class KeplersThirdLawSymbolic(RulePlugin):
    id = "keplers_third_law_symbolic"
    title = "Symbolic Check for Kepler's Third Law"
    description = "Checks physically if T^2 is proportional to r^3 using symbolic graph analysis."

    def run(self, ctx: RuleContext, rt: RuleRuntime) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []

        sem_graph = getattr(ctx, "semantic_graph", None)
        if sem_graph is None:
            sem_graph = EnrichedSymbolGraph(ctx.graph)
            ctx.semantic_graph = sem_graph

        period_syms = [s for s in ctx.graph.symbols if s in ["T", "P", "T_orbit"]]
        radius_syms = [s for s in ctx.graph.symbols if s in ["r", "R", "a", "d"]]

        if not period_syms or not radius_syms:
            return []

        for p_sym in period_syms:
            for r_sym in radius_syms:
                findings = sem_graph.get_relationship(p_sym, r_sym)
                for finding in findings:
                    exponent = finding.get("exponent")
                    equation = finding.get("equation")
                    fid = finding.get("fid")

                    if exponent is None:
                        continue

                    if abs(exponent - 1.5) < 0.1:
                        continue
                    if abs(exponent - 1.0) < 0.1:
                        diagnostics.append(
                            {
                                "severity": "error",
                                "rule": self.id,
                                "symbol": fid,
                                "message": (
                                    "Violation of Kepler's 3rd Law: Found linear relationship "
                                    f"(T ~ r^1.0) instead of T ~ r^1.5 in {equation}"
                                ),
                                "evidence": ctx.graph.formulas[fid].raw if fid in ctx.graph.formulas else None,
                            }
                        )
                    elif abs(exponent - 0.5) < 0.1:
                        diagnostics.append(
                            {
                                "severity": "error",
                                "rule": self.id,
                                "symbol": fid,
                                "message": (
                                    "Violation of Kepler's 3rd Law: Found sqrt relationship "
                                    f"(T ~ r^0.5) instead of T ~ r^1.5 in {equation}"
                                ),
                                "evidence": ctx.graph.formulas[fid].raw if fid in ctx.graph.formulas else None,
                            }
                        )

        return diagnostics


class LatexSyntaxSymbolic(RulePlugin):
    id = "latex_syntax_symbolic"
    title = "Latex Syntax Checker"
    description = "Checks for unbalanced braces in LaTeX fragments."

    def run(self, ctx: RuleContext, rt: RuleRuntime) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []
        for fid, formula in ctx.graph.formulas.items():
            raw = formula.raw
            open_braces = raw.count("{")
            close_braces = raw.count("}")
            if open_braces != close_braces:
                diagnostics.append(
                    {
                        "severity": "error",
                        "rule": self.id,
                        "symbol": fid,
                        "message": f"Unbalanced LaTeX braces: {open_braces} '{{' vs {close_braces} '}}'",
                        "evidence": raw,
                    }
                )
        return diagnostics


class TimeDilationLengthContractionSymbolic(RulePlugin):
    id = "time_dilation_length_contraction_symbolic"
    title = "Symbolic Check for Special Relativity Formulas"
    description = "Checks correct application of gamma factor in Time Dilation vs Length Contraction."

    def run(self, ctx: RuleContext, rt: RuleRuntime) -> List[Dict[str, Any]]:
        diagnostics: List[Dict[str, Any]] = []

        sem_graph = getattr(ctx, "semantic_graph", None)
        if sem_graph is None:
            sem_graph = EnrichedSymbolGraph(ctx.graph)
            ctx.semantic_graph = sem_graph

        proper_length = ["L_0", "l_0", "L0", "d_0", "x_0", "L_{0}", "l_{0}", "d_{0}", "x_{0}"]
        rel_length = ["L", "l", "L'", "d", "x"]
        proper_time = ["t_0", "tau", r"\tau", "T_0", "t_{0}", "T_{0}"]
        rel_time = ["t", "T", "t'"]

        v_sym = sympy.Symbol("v")
        c_sym = sympy.Symbol("c")

        for fid, expr in sem_graph.parsed_formulas.items():
            fsyms = {s.name for s in expr.free_symbols}

            p_time = next((s for s in proper_time if s in fsyms), None)
            r_time = next((s for s in rel_time if s in fsyms), None)

            if p_time and r_time:
                findings = sem_graph.get_relationship(r_time, p_time)
                for finding in findings:
                    rhs = finding.get("rhs")
                    if rhs is None:
                        continue
                    if v_sym in rhs.free_symbols and c_sym in rhs.free_symbols:
                        try:
                            val = rhs.subs({sympy.Symbol(p_time): 1.0, v_sym: 0.5, c_sym: 1.0}).evalf()
                            if float(val) < 0.99:
                                diagnostics.append(
                                    {
                                        "severity": "error",
                                        "rule": self.id,
                                        "symbol": fid,
                                        "message": (
                                            "Incorrect Time Dilation: relativistic time < proper time "
                                            f"(factor {float(val):.2f}); expected factor > 1."
                                        ),
                                        "evidence": ctx.graph.formulas[fid].raw if fid in ctx.graph.formulas else None,
                                    }
                                )
                        except Exception:
                            pass

            p_len = next((s for s in proper_length if s in fsyms), None)
            r_len = next((s for s in rel_length if s in fsyms), None)
            if p_len and r_len:
                findings = sem_graph.get_relationship(r_len, p_len)
                for finding in findings:
                    rhs = finding.get("rhs")
                    if rhs is None:
                        continue
                    if v_sym in rhs.free_symbols and c_sym in rhs.free_symbols:
                        try:
                            val = rhs.subs({sympy.Symbol(p_len): 1.0, v_sym: 0.5, c_sym: 1.0}).evalf()
                            if float(val) > 1.01:
                                diagnostics.append(
                                    {
                                        "severity": "error",
                                        "rule": self.id,
                                        "symbol": fid,
                                        "message": (
                                            "Incorrect Length Contraction: relativistic length > proper length "
                                            f"(factor {float(val):.2f}); expected factor < 1."
                                        ),
                                        "evidence": ctx.graph.formulas[fid].raw if fid in ctx.graph.formulas else None,
                                    }
                                )
                        except Exception:
                            pass

        return diagnostics


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

    SUPPORTED_PRIMITIVES = {"power_law", "equation_equivalence"}

    def run(self, ctx: RuleContext, specs: List[GeneratedSymbolicCheckSpec]) -> List[Dict[str, Any]]:
        diags: List[Dict[str, Any]] = []

        sem_graph = getattr(ctx, "semantic_graph", None)
        if sem_graph is None:
            sem_graph = EnrichedSymbolGraph(ctx.graph)
            ctx.semantic_graph = sem_graph

        for spec in specs:
            if spec.primitive not in self.SUPPORTED_PRIMITIVES:
                continue

            if spec.primitive == "power_law":
                diags.extend(self._run_power_law(ctx, sem_graph, spec))
            elif spec.primitive == "equation_equivalence":
                diags.extend(self._run_equation_equivalence(ctx, sem_graph, spec))

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

        out: List[Dict[str, Any]] = []
        any_relationship_found = False
        for dep in dependent_candidates:
            for indep in independent_candidates:
                findings = sem_graph.get_relationship(dep, indep)
                for finding in findings:
                    any_relationship_found = True
                    exponent = finding.get("exponent")
                    fid = finding.get("fid")
                    if exponent is None or fid is None:
                        continue
                    derived = float(exponent) * float(dependent_power)
                    if abs(derived - expected_exponent) <= tolerance:
                        continue
                    out.append(
                        {
                            "severity": "warning",
                            "rule": f"agentic_symbolic::{spec.spec_id}",
                            "spec_id": spec.spec_id,
                            "symbolic_result": "fail",
                            "symbol": fid,
                            "message": (
                                f"Symbolic cross-check failed for '{spec.title}': expected {dep}^{dependent_power:g} ~ {indep}^{expected_exponent} "
                                f"(±{tolerance}), found {dep} ~ {indep}^{float(exponent):.2f} (=> {dep}^{dependent_power:g} ~ {indep}^{derived:.2f})."
                            ),
                            "evidence": ctx.graph.formulas[fid].raw if fid in ctx.graph.formulas else None,
                        }
                    )

        # If we couldn't find any equation that relates the candidates, we can't refute/confirm the original diagnostic.
        if not any_relationship_found:
            return [
                {
                    "severity": "info",
                    "rule": f"agentic_symbolic::{spec.spec_id}",
                    "spec_id": spec.spec_id,
                    "symbolic_result": "inconclusive",
                    "symbol": None,
                    "message": (
                        f"Symbolic cross-check inconclusive for '{spec.title}': no equation relates the provided variables "
                        f"(dependent={dependent_candidates}, independent={independent_candidates})."
                    ),
                    "evidence": None,
                }
            ]
        return out

    def _run_equation_equivalence(
        self, ctx: RuleContext, sem_graph: EnrichedSymbolGraph, spec: GeneratedSymbolicCheckSpec
    ) -> List[Dict[str, Any]]:
        """Check whether any extracted equation is algebraically equivalent to a canonical equation (up to scalar factor)."""

        params = spec.params or {}
        canonical_latex_list = params.get("canonical_latex") or []
        required_symbols = params.get("required_symbols") or []
        allow_scalar_multiple = bool(params.get("allow_scalar_multiple", True))

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
            return [
                {
                    "severity": "info",
                    "rule": f"agentic_symbolic::{spec.spec_id}",
                    "spec_id": spec.spec_id,
                    "symbolic_result": "inconclusive",
                    "symbol": None,
                    "message": f"Symbolic cross-check inconclusive for '{spec.title}': canonical equations could not be parsed.",
                    "evidence": None,
                }
            ]

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
                    return []

        # We found relevant equations but none matches the canonical ones.
        fid0, expr0 = candidates[0]
        return [
            {
                "severity": "warning",
                "rule": f"agentic_symbolic::{spec.spec_id}",
                "spec_id": spec.spec_id,
                "symbolic_result": "fail",
                "symbol": fid0,
                "message": f"Symbolic cross-check failed for '{spec.title}': no extracted equation is equivalent to the canonical form.",
                "evidence": ctx.graph.formulas[fid0].raw if fid0 in ctx.graph.formulas else None,
            }
        ]


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
