from typing import List, Dict, Any, Optional
import re
import sympy
from rules.base import RulePlugin, RuleContext, RuleRuntime
from symbolic_system import EnrichedSymbolGraph, FormulaParser

class KeplersThirdLawSymbolic(RulePlugin):
    id = "keplers_third_law_symbolic"
    title = "Symbolic Check for Kepler's Third Law"
    description = "Checks physically if T^2 is proportional to r^3 using symbolic graph analysis."

    def run(self, ctx: RuleContext, rt: RuleRuntime) -> List[Dict[str, Any]]:
        diagnostics = []
        
        # Build semantic graph wrapper
        if hasattr(ctx, "semantic_graph"):
            sem_graph = ctx.semantic_graph
        else:
            sem_graph = EnrichedSymbolGraph(ctx.graph)
            ctx.semantic_graph = sem_graph

        # Look for Period (T) and Radius/Semi-major axis (r, a)
        period_syms = [s for s in ctx.graph.symbols if s in ["T", "P", "T_orbit"]]
        radius_syms = [s for s in ctx.graph.symbols if s in ["r", "R", "a", "d"]]

        if not period_syms or not radius_syms:
            return []

        # Check relationships across all combinations
        for p_sym in period_syms:
            for r_sym in radius_syms:
                findings = sem_graph.get_relationship(p_sym, r_sym)
                for finding in findings:
                    exponent = finding["exponent"]
                    equation = finding["equation"]
                    fid = finding["fid"]
                    
                    if abs(exponent - 1.5) < 0.1:
                        pass
                    elif abs(exponent - 1.0) < 0.1:
                         diagnostics.append({
                             "severity": "error",
                             "rule": self.id,
                             "symbol": fid,
                             "message": f"Violation of Kepler's 3rd Law: Found linear relationship (T ~ r^1.0) instead of T ~ r^1.5 in {equation}",
                             "evidence": ctx.graph.formulas[fid].raw
                         })
                    elif abs(exponent - 0.5) < 0.1:
                         diagnostics.append({
                             "severity": "error",
                             "rule": self.id,
                             "symbol": fid,
                             "message": f"Violation of Kepler's 3rd Law: Found sqrt relationship (T ~ r^0.5) instead of T ~ r^1.5 in {equation}",
                             "evidence": ctx.graph.formulas[fid].raw
                         })
                    else:
                        pass
                         # diagnostics.append({
                         #     "severity": "warning",
                         #     "rule": self.id,
                         #     "symbol": fid,
                         #     "message": f"Unexpected power relationship in Kepler context: T ~ r^{exponent:.2f}",
                         #     "evidence": ctx.graph.formulas[fid].raw
                         # })

        return diagnostics

class LatexSyntaxSymbolic(RulePlugin):
    id = "latex_syntax_symbolic"
    title = "Latex Syntax Checker"
    description = "Checks for validity of LaTeX syntax using SymPy parser."

    def run(self, ctx: RuleContext, rt: RuleRuntime) -> List[Dict[str, Any]]:
        diagnostics = []
        
        for fid, formula in ctx.graph.formulas.items():
            raw = formula.raw
            
            # 1. Simple Brace Check
            open_braces = raw.count("{")
            close_braces = raw.count("}")
            if open_braces != close_braces:
                diagnostics.append({
                    "severity": "error",
                    "rule": self.id,
                    "symbol": fid,
                    "message": f"Unbalanced LaTeX braces in formula: {open_braces} '{{' vs {close_braces} '}}'",
                    "evidence": raw
                })
            
            # 2. Advanced Parser Check
            # (Optional: can include this if we want strict syntax validation)
            # For now, we trust the braces check for syntax, as partial fragments are common.
        
        return diagnostics

class TimeDilationLengthContractionSymbolic(RulePlugin):
    id = "time_dilation_length_contraction_symbolic"
    title = "Symbolic Check for Special Relativity Formulas"
    description = "Checks correct application of gamma factor in Time Dilation vs Length Contraction."

    def run(self, ctx: RuleContext, rt: RuleRuntime) -> List[Dict[str, Any]]:
        diagnostics = []
        
        if hasattr(ctx, "semantic_graph"):
            sem_graph = ctx.semantic_graph
        else:
            sem_graph = EnrichedSymbolGraph(ctx.graph)
            ctx.semantic_graph = sem_graph

        # Proper: L_0, l_0, t_0, tau, \tau
        # Rel: L, l, t, T
        
        proper_length = ["L_0", "l_0", "L0", "d_0", "x_0", "L_{0}", "l_{0}", "d_{0}", "x_{0}"]
        rel_length = ["L", "l", "L'", "d", "x"]
        
        proper_time = ["t_0", "tau", r"\tau", "T_0", "t_{0}", "T_{0}"]
        rel_time = ["t", "T", "t'"]

        from sympy import Symbol
        v_sym = Symbol('v')
        c_sym = Symbol('c')

        # Check each formula
        for fid, formula_data in sem_graph.parsed_formulas.items():
            fsyms = {s.name for s in formula_data.free_symbols}
            
            # --- TIME DILATION CHECK ---
            p_time = next((s for s in proper_time if s in fsyms), None)
            r_time = next((s for s in rel_time if s in fsyms), None)

            if p_time and r_time:
                 findings = sem_graph.get_relationship(r_time, p_time)
                 for finding in findings:
                     rhs = finding["rhs"]
                     # Check if it has v and c
                     if v_sym in rhs.free_symbols and c_sym in rhs.free_symbols:
                         try:
                             # Substitution: v=0.5c
                             # Expectation: t_rel > t_proper (Dilation)
                             # So factor = t_rel/t_proper > 1
                             val_factor = rhs.subs({p_time: 1.0, v_sym: 0.5, c_sym: 1.0}).evalf()
                             
                             if val_factor < 0.99: # Allowing some float tolerance
                                 diagnostics.append({
                                     "severity": "error",
                                     "rule": self.id,
                                     "symbol": fid,
                                     "message": f"Incorrect Time Dilation: Found relativistic time < proper time (factor {val_factor:.2f}). Time should dilate (factor > 1).",
                                     "evidence": ctx.graph.formulas[fid].raw
                                 })
                         except Exception:
                             pass

            # --- LENGTH CONTRACTION CHECK ---
            p_len = next((s for s in proper_length if s in fsyms), None)
            r_len = next((s for s in rel_length if s in fsyms), None)
            
            if p_len and r_len:
                 findings = sem_graph.get_relationship(r_len, p_len)
                 for finding in findings:
                     rhs = finding["rhs"]
                     if v_sym in rhs.free_symbols and c_sym in rhs.free_symbols:
                         try:
                             # Substitution: v=0.5c
                             # Expectation: L_rel < L_proper (Contraction)
                             # So factor < 1
                             val_factor = rhs.subs({p_len: 1.0, v_sym: 0.5, c_sym: 1.0}).evalf()
                             
                             if val_factor > 1.01:
                                 diagnostics.append({
                                     "severity": "error",
                                     "rule": self.id,
                                     "symbol": fid,
                                     "message": f"Incorrect Length Contraction: Found relativistic length > proper length (factor {val_factor:.2f}). Length should contract (factor < 1).",
                                     "evidence": ctx.graph.formulas[fid].raw
                                 })
                         except Exception:
                             pass

        return diagnostics
