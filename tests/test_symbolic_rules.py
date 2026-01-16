import json
import sys
import os
from pathlib import Path

# Add project root to path (parent of tests dir)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rule_based_verifier import RuleBasedVerifier, SymbolGraph, RuleContext
from rules.symbolic_checks import (
    KeplersThirdLawSymbolic,
    LatexSyntaxSymbolic,
    TimeDilationLengthContractionSymbolic,
    GeneratedSymbolicCheckExecutor,
    GeneratedSymbolicCheckSpec,
)

def run_tests():
    # Load data (relative to project root if running from root, or relative to script?)
    # Assuming run from project root: data/evaluation_sample_30.json
    # If run from inside tests/, we might need ../data
    
    # Robust path finding
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = base_dir / "data/evaluation_sample_30.json"
    
    if not data_path.exists():
        print("Data file not found")
        return

    with open(data_path, "r") as f:
        samples = json.load(f)
    
    # Map id to sample, but handle duplicates by keeping the one with specific content
    target_iss_id = "167622"
    iss_sample = None
    
    for s in samples:
        if s["id"] == target_iss_id and "International Space Station" in s["question"]:
            iss_sample = s
            break
            
    # Initialize verifier to use its graph building capability
    verifier = RuleBasedVerifier(max_llm_calls=0)
    
    # Test 1: Kepler's Third Law & Latex Syntax on Sample 167622 (ISS)
    if iss_sample:
        print(f"--- Testing Sample {target_iss_id} (ISS) ---")
        sample = iss_sample
        text_all = sample["question"] + "\n" + sample["prediction"]
        
        # Build graph
        parsed = verifier._extract_symbols_and_formulas(text_all)
        graph = verifier._build_symbol_graph(parsed["lines"], parsed["symbols"], parsed["formulas"])
        
        ctx = RuleContext(
            sample_id=target_iss_id,
            dataset_key=None,
            text_all=text_all,
            lines=parsed["lines"],
            symbols=parsed["symbols"],
            formulas_raw=parsed["formulas"],
            graph=graph,
            snippets={},
            sym_stats={},
            precondition_cues=[]
        )
        
        # Run Latex Check
        rule_latex = LatexSyntaxSymbolic()
        diags = rule_latex.run(ctx, None)
        if diags:
            print(f"[{rule_latex.id}] Found issues:")
            for d in diags:
                print(f"  - {d['message']} (Evidence: {d['evidence']})")
        else:
            print(f"[{rule_latex.id}] Passed")

        # Run Kepler Check
        rule_kepler = KeplersThirdLawSymbolic()
        diags = rule_kepler.run(ctx, None)
        if diags:
            print(f"[{rule_kepler.id}] Found issues:")
            for d in diags:
                print(f"  - {d['message']} (Evidence: {d['evidence']})")
        else:
            print(f"[{rule_kepler.id}] Passed (No incorrect linear dependence found)")

        # Regression: agentic power_law spec for Kepler should NOT fail when T ~ r^(3/2)
        exec_ = GeneratedSymbolicCheckExecutor()
        kepler_spec = GeneratedSymbolicCheckSpec(
            spec_id="test_kepler_power_law_dep_power",
            title="Kepler T^2 ~ r^3 regression",
            description="Ensure dependent_power handling matches Kepler's 3rd law.",
            primitive="power_law",
            params={
                "dependent_candidates": ["T"],
                "independent_candidates": ["r"],
                "dependent_power": 2,
                "expected_exponent": 3,
                "tolerance": 0.2,
            },
            source_rule_id="keplers_third_law_check_01",
            source_message_substring="regression",
        )
        diags = exec_.run(ctx, [kepler_spec])
        if diags:
            print("[agentic_kepler_power_law] Unexpected FAIL:")
            for d in diags:
                print(f"  - {d['message']}")
        else:
            print("[agentic_kepler_power_law] Passed")

        # Regression: equation_equivalence should accept standard Kepler form present in the sample.
        kepler_eq_spec = GeneratedSymbolicCheckSpec(
            spec_id="test_kepler_equiv",
            title="Kepler equation equivalence regression",
            description="Ensure equation_equivalence matches canonical Kepler forms.",
            primitive="equation_equivalence",
            params={
                "canonical_latex": [
                    "T = 2\\pi \\sqrt{\\frac{r^3}{G M_E}}",
                    "T^2 = \\frac{4\\pi^2}{G M_E} r^3",
                ],
                "required_symbols": ["T", "r"],
                "allow_scalar_multiple": False,
            },
            source_rule_id="keplers_third_law_check_01",
            source_message_substring="regression",
        )
        diags = exec_.run(ctx, [kepler_eq_spec])
        if diags:
            print("[agentic_kepler_equiv] Unexpected FAIL:")
            for d in diags:
                print(f"  - {d['message']}")
        else:
            print("[agentic_kepler_equiv] Passed")

    # Test 2: Time Dilation check on Sample 29185
    sid = "29185"
    sample_29185 = next((s for s in samples if s["id"] == sid), None)
    
    if sample_29185:
        print(f"\n--- Testing Sample {sid} ---")
        sample = sample_29185
        text_all = sample["question"] + "\n" + sample["prediction"]
        
        # Build graph
        parsed = verifier._extract_symbols_and_formulas(text_all)
        graph = verifier._build_symbol_graph(parsed["lines"], parsed["symbols"], parsed["formulas"])
        
        ctx = RuleContext(
            sample_id=sid,
            dataset_key=None,
            text_all=text_all,
            lines=parsed["lines"],
            symbols=parsed["symbols"],
            formulas_raw=parsed["formulas"],
            graph=graph,
            snippets={},
            sym_stats={},
            precondition_cues=[]
        )
        
        # Run Time Dilation Check
        rule_relativity = TimeDilationLengthContractionSymbolic()
        diags = rule_relativity.run(ctx, None)
        if diags:
            print(f"[{rule_relativity.id}] Found issues:")
            for d in diags:
                print(f"  - {d['message']} (Evidence: {d['evidence']})")
        else:
            print(f"[{rule_relativity.id}] Passed")

if __name__ == "__main__":
    run_tests()
