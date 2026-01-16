import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from core.top_down_verifier import TopDownVerifier


def _strip_symbolic_fields_from_diagnostic(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    out.pop("symbolic_cross_checks", None)
    out.pop("symbolic_reconciliation", None)
    return out


def _build_main_result(sample_result: Dict[str, Any]) -> Dict[str, Any]:
    diagnostics = []
    for d in sample_result.get("diagnostics", []) or []:
        if isinstance(d, dict):
            diagnostics.append(_strip_symbolic_fields_from_diagnostic(d))

    return {
        "id": sample_result.get("id"),
        "topic": sample_result.get("topic"),
        "verifier": sample_result.get("verifier"),
        "diagnostics": diagnostics,
        "score": sample_result.get("score"),
    }


def _build_symbolic_audit(sample_result: Dict[str, Any]) -> Dict[str, Any]:
    checked: List[Dict[str, Any]] = []
    for d in sample_result.get("diagnostics", []) or []:
        if not isinstance(d, dict):
            continue
        if d.get("symbolic_cross_checks") or d.get("symbolic_reconciliation"):
            checked.append(
                {
                    "severity": d.get("severity"),
                    "rule": d.get("rule"),
                    "symbol": d.get("symbol"),
                    "message": d.get("message"),
                    "evidence": d.get("evidence"),
                    "symbolic_cross_checks": d.get("symbolic_cross_checks") or [],
                    "symbolic_reconciliation": d.get("symbolic_reconciliation"),
                }
            )

    agentic = sample_result.get("agentic") if isinstance(sample_result.get("agentic"), dict) else {}

    return {
        "id": sample_result.get("id"),
        "topic": sample_result.get("topic"),
        "checked_diagnostics": checked,
        "symbolic_post_diagnostics": sample_result.get("symbolic_post_diagnostics", []) or [],
        "suppressed_diagnostics": (agentic.get("suppressed_diagnostics") or []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PhysicsVerifier top-down checking.")
    parser.add_argument("--input", "-i", type=str, default="data/evaluation_sample_30.json")
    parser.add_argument("--output", "-o", type=str, default="results/top_down_results.json")
    parser.add_argument(
        "--symbolic-output",
        type=str,
        default="results/symbolic_audit.json",
        help="Write symbolic cross-check audit (checked diagnostics + symbolic results) to this file.",
    )
    parser.add_argument("--catalog", type=str, default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--model", type=str, default="qwen3-30b-a3b")
    parser.add_argument("--no-agentic", action="store_true")
    parser.add_argument("--agentic-max", type=int, default=2)

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    verifier = TopDownVerifier(
        rules_catalog_path=args.catalog,
        llm_model=args.model,
        enable_agentic_postcheck=not args.no_agentic,
        agentic_max_checks_per_sample=args.agentic_max,
    )

    raw_results = verifier.run_batch(samples)

    # Main output: only final diagnostics (after symbolic suppression), without symbolic metadata.
    results = [_build_main_result(r) for r in (raw_results or []) if isinstance(r, dict)]

    # Symbolic audit output: only samples that had symbolic checks or produced symbolic outputs.
    symbolic_audit_all = [_build_symbolic_audit(r) for r in (raw_results or []) if isinstance(r, dict)]
    symbolic_audit = [
        a
        for a in symbolic_audit_all
        if (a.get("checked_diagnostics") or a.get("symbolic_post_diagnostics") or a.get("suppressed_diagnostics"))
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    sym_path = Path(args.symbolic_output)
    sym_path.parent.mkdir(parents=True, exist_ok=True)
    sym_path.write_text(json.dumps(symbolic_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Results saved to {out_path}")
    print(f"Done. Symbolic audit saved to {sym_path}")


if __name__ == "__main__":
    main()
