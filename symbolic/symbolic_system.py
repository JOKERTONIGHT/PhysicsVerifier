import sympy
from sympy.parsing.latex import parse_latex
from sympy import Symbol, Eq, solve, simplify
from typing import Dict, List, Any, Optional, Tuple

class FormulaParser:
    """
    Parses LaTeX strings into SymPy expressions.
    Handles basic cleaning and robustness.
    """
    @staticmethod
    def parse(latex_str: str) -> Optional[sympy.Expr]:
        cleaned = latex_str.strip()
        # Remove common non-math markers if any (though graph extraction should have handled most)
        # Basic cleaning: remove \mathrm{}, \text{} wrappers if they wrap single vars, 
        # but parse_latex handles some of this.
        
        # Remove trailing punctuation often captured like "." or ","
        if cleaned.endswith('.') or cleaned.endswith(','):
            cleaned = cleaned[:-1]
            
        try:
            # Attempt direct parsing
            # parse_latex returns a SymPy expression. 
            # If it's an equation "a=b", it might return an Eq object or a relation.
            expr = parse_latex(cleaned)
            return expr
        except Exception:
            # Fallback or specific handling could go here
            return None

class RelationAnalyzer:
    """
    Analyzes mathematical relationships between symbols in equations.
    """
    @staticmethod
    def check_power_relationship(eq: sympy.Expr, dependent_var: str, independent_var: str) -> Optional[float]:
        """
        Checks if 'dependent_var' scales with 'independent_var' to some power.
        Returns the exponent if a clean power law is found, else None.
        e.g., T = k * r^1.5 -> returns 1.5
        """
        try:
            dep_sym = Symbol(dependent_var)
            indep_sym = Symbol(independent_var)
            
            # If the equation is an Equality, Try to solve for dependent_var
            # Note: parse_latex might return Eq(lhs, rhs) or just an expression (implicitly = 0 or just an expression)
            
            if isinstance(eq, sympy.Eq):
                solutions = solve(eq, dep_sym)
                if not solutions:
                    return None
                # Take the first solution (physics problems usually implies positive real roots for these quantities)
                sol = solutions[0]
            else:
                # Assume expression equals 0
                solutions = solve(eq, dep_sym)
                if not solutions:
                    return None
                sol = solutions[0]

            # Analyze the dependence on indep_sym
            # We can differentiate log(sol) / log(indep_sym) or just inspect properties
            # Simple approach: compute exponent if term is monomial-like
            # Or: substitutions
            
            # Method: let all other symbols be 1, let indep_sym be x. Check form x^p.
            # Identify other symbols
            free_syms = sol.free_symbols
            subs_dict = {s: 1 for s in free_syms if s.name != independent_var}
            
            simplified_sol = sol.subs(subs_dict)
            
            # Match strict power law: C * x^p
            # log(simplified_sol) should be log(C) + p*log(x)
            # diff(log(simplified_sol), x) * x should be p
            
            x = Symbol('x_dummy')
            expr_x = simplified_sol.replace(indep_sym, x)
            
            # Check scaling: ratio test?
            # f(k*x) / f(x) = k^p
            k = sympy.Number(2)
            ratio = simplify(expr_x.subs(x, k*x) / expr_x)
            
            # If ratio is a number, we have a power law. 
            # 2^p = ratio => p = log2(ratio)
            if ratio.is_number:
                p = sympy.log(ratio, 2)
                return float(p)
            
            return None

        except Exception as e:
            # Parsing/solving failed
            return None

    @staticmethod
    def check_factor_influence(eq: sympy.Expr, target_var: str, factor_fingerprint: str) -> str:
        """
        Determines if a 'factor' (identified by string fingerprint in raw latex?) 
        behaves as a multiplier or divisor to the target_var.
        
        Since we have SymPy expr, we might look for specific sub-expressions.
        However, mapping "sqrt(1-v^2/c^2)" back to a symbol is tricky if not parsed as one.
        
        Alternative: The caller provides the SymPy symbol representing the factor.
        We check if target ~ C * factor or target ~ C / factor.
        """
        return "complex"

class EnrichedSymbolGraph:
    """
    Wraps the raw dictionary-based SymbolGraph with Semantic information.
    """
    def __init__(self, raw_graph: Any):
        self.raw = raw_graph
        self.parsed_formulas: Dict[str, sympy.Expr] = {}
        self.parsing_errors: Dict[str, str] = {}
        
        self._parse_all()
        
    def _parse_all(self):
        for fid, f_node in self.raw.formulas.items():
            raw_latex = f_node.raw
            parsed = FormulaParser.parse(raw_latex)
            if parsed is not None:
                self.parsed_formulas[fid] = parsed
            else:
                self.parsing_errors[fid] = "Failed to parse LaTeX"

    def get_equations_involving(self, symbols: List[str]) -> Dict[str, sympy.Expr]:
        """
        Return all parsed equations that contain AT LEAST ONE of the provided symbols.
        Or maybe ALL? Let's say at least one for loose search, but usually we want relations between specific vars.
        """
        results = {}
        target_set = set(symbols)
        for fid, expr in self.parsed_formulas.items():
            # free_symbols returns SymPy symbols. Convert to string to compare.
            expr_syms = {s.name for s in expr.free_symbols}
            
            # Check intersection
            # Note: SymPy parsing might mangle names slightly (e.g. latin/greek), but usually preserves basic names.
            # A robust system would map names. Here we assume direct mapping.
            if not expr_syms.isdisjoint(target_set):
                results[fid] = expr
        return results

    def get_relationship(self, dependent: str, independent: str) -> List[Dict[str, Any]]:
        """
        Finds all equations relating these two variables and analyzes the relationship.
        """
        # Find equations containing BOTH
        candidates = {}
        for fid, expr in self.parsed_formulas.items():
            syms = {s.name for s in expr.free_symbols}
            if dependent in syms and independent in syms:
                candidates[fid] = expr
        
        findings = []
        for fid, eq in candidates.items():
            power = RelationAnalyzer.check_power_relationship(eq, dependent, independent)
            
            # Solve for dependent var to allow coefficient analysis
            solved_rhs = None
            try:
                dep_sym = sympy.Symbol(dependent)
                # If eq is Equality, solve implies eq.lhs - eq.rhs = 0
                solutions = sympy.solve(eq, dep_sym)
                if solutions:
                    solved_rhs = solutions[0]
            except:
                pass

            if power is not None:
                findings.append({
                    "fid": fid,
                    "equation": eq,
                    "type": "power_law",
                    "exponent": power,
                    "rhs": solved_rhs
                })
        return findings

