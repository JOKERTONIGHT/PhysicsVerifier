import os
import sys
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


# Ensure project root is on sys.path so `import core.*` works when this file is executed
# as a script from arbitrary working directories.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.physics_rule_verifier import PhysicsRuleVerifier


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
            recon = d.get("symbolic_reconciliation") or {}
            checked.append(
                {
                    "severity": d.get("severity"),
                    "rule": d.get("rule"),
                    "symbol": d.get("symbol"),
                    "message": d.get("message"),
                    "evidence": d.get("evidence"),
                    "rule_match": d.get("rule_match"),
                    "release_gate": d.get("release_gate"),
                    "symbolic_cross_checks": d.get("symbolic_cross_checks") or [],
                    "symbolic_reconciliation": d.get("symbolic_reconciliation"),
                    "symbolic_status": recon.get("status"),
                }
            )

    agentic = sample_result.get("agentic") if isinstance(sample_result.get("agentic"), dict) else {}

    symbolic_checks = []
    for sd in sample_result.get("symbolic_post_diagnostics", []) or []:
        if not isinstance(sd, dict):
            continue
        symbolic_checks.append(
            {
                "spec_id": sd.get("spec_id"),
                "primitive": sd.get("primitive"),
                "title": sd.get("title"),
                "result": sd.get("symbolic_result"),
                "rule": sd.get("rule"),
                "symbol": sd.get("symbol"),
                "message": sd.get("message"),
                "evidence": sd.get("evidence"),
                "details": sd.get("details"),
            }
        )

    summary = {
        "total": len(symbolic_checks),
        "pass": len([c for c in symbolic_checks if c.get("result") == "pass"]),
        "fail": len([c for c in symbolic_checks if c.get("result") == "fail"]),
        "inconclusive": len([c for c in symbolic_checks if c.get("result") == "inconclusive"]),
    }

    experience_checks = []
    for ed in sample_result.get("experience_post_diagnostics", []) or []:
        if not isinstance(ed, dict):
            continue
        experience_checks.append(
            {
                "rule": ed.get("rule"),
                "message": ed.get("message"),
                "evidence": ed.get("evidence"),
                "symbolic_cross_checks": ed.get("experience_symbolic_cross_checks") or [],
                "symbolic_reconciliation": ed.get("experience_symbolic_reconciliation"),
            }
        )

    experience_symbolic_checks = []
    for sd in sample_result.get("experience_symbolic_post_diagnostics", []) or []:
        if not isinstance(sd, dict):
            continue
        experience_symbolic_checks.append(
            {
                "spec_id": sd.get("spec_id"),
                "primitive": sd.get("primitive"),
                "title": sd.get("title"),
                "result": sd.get("symbolic_result"),
                "rule": sd.get("rule"),
                "symbol": sd.get("symbol"),
                "message": sd.get("message"),
                "evidence": sd.get("evidence"),
            }
        )

    experience_code_checks = []
    for cd in sample_result.get("experience_code_post_diagnostics", []) or []:
        if not isinstance(cd, dict):
            continue
        experience_code_checks.append(
            {
                "rule": cd.get("rule"),
                "rule_id": cd.get("rule_id"),
                "result": cd.get("result"),
                "symbolic_result": cd.get("symbolic_result"),
                "source": cd.get("source"),
                "bridge_for_rule_id": cd.get("bridge_for_rule_id"),
                "publish_skipped": cd.get("publish_skipped"),
                "message": cd.get("message"),
                "evidence": cd.get("evidence"),
            }
        )

    return {
        "id": sample_result.get("id"),
        "topic": sample_result.get("topic"),
        "candidate_diagnostic_count": len(sample_result.get("candidate_diagnostics") or []),
        "checked_diagnostics": checked,
        "symbolic_summary": summary,
        "symbolic_checks": symbolic_checks,
        "experience_checks": experience_checks,
        "experience_symbolic_checks": experience_symbolic_checks,
        "experience_code_checks": experience_code_checks,
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
    parser.add_argument(
        "--full-output",
        type=str,
        default="",
        help="Optional path to write full raw verifier outputs (includes retrieved topics/rules and agentic payload).",
    )
    parser.add_argument("--catalog", type=str, default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--model", type=str, default="qwen3-30b-a3b")
    parser.add_argument(
        "--no-symbolic-check",
        action="store_true",
        help="Disable the deterministic experience-code symbolic verification (on by default).",
    )
    parser.add_argument(
        "--unified-catalog",
        type=str,
        default=None,
        help="Path to the unified rules catalog JSON. When set and the file exists, "
             "this takes priority over --catalog.",
    )
    parser.add_argument(
        "--experience-code-manifest",
        type=str,
        default="results/experience_symbolic_program_manifest_v2_unified.json",
        help="Manifest produced by scripts/generate_symbolic_checks.py mapping rule_id -> Python check function.",
    )
    parser.add_argument(
        "--experience-code-module",
        type=str,
        default="symbolic.generated_experience_checks_v2_unified",
        help="Python module path containing the generated experience-code check functions.",
    )
    parser.add_argument(
        "--symbolic-topic-check-limit",
        type=int,
        default=40,
        help="Max bottom-up experience-code checks to run per retrieved topic (<=0 disables the cap).",
    )
    # Legacy flags (accepted for backward compatibility, ignored).
    parser.add_argument("--no-agentic", action="store_true", help="(deprecated, no-op)")
    parser.add_argument("--agentic-max", type=int, default=2, help="(deprecated, no-op)")
    parser.add_argument(
        "--experience",
        action="store_true",
        help="(deprecated, no-op; experience-code symbolic check now runs by default)",
    )
    parser.add_argument(
        "--experience-rules",
        type=str,
        default=None,
        help="(deprecated, no-op; rule library is taken from --unified-catalog)",
    )
    parser.add_argument(
        "--precision-mode",
        type=str,
        default="strict",
        choices=["strict", "balanced", "score_only"],
        help="Diagnostic publication policy for unified v2 rules.",
    )
    parser.add_argument(
        "--min-diagnostic-rule-score",
        type=float,
        default=None,
        help="Override the minimum rule score for diagnostic publication/sweeps.",
    )
    parser.add_argument(
        "--unified-rule-top-n",
        type=int,
        default=None,
        help="Unified v2: maximum rules to retrieve per sample for SRD injection (default 6).",
    )
    parser.add_argument(
        "--topic-skip-prediction",
        action="store_true",
        help="For unified v2, exclude prediction from topic retrieval text (sets PHYSICSVERIFIER_TOPIC_SKIP_PREDICTION).",
    )
    parser.add_argument(
        "--max-per-sample",
        type=int,
        default=None,
        help="Maximum diagnostics published per sample (precision cap). Set <=0 to disable.",
    )
    parser.add_argument(
        "--max-per-paragraph",
        type=int,
        default=None,
        help="Maximum diagnostics published per paragraph within a sample. Set <=0 to disable.",
    )
    parser.add_argument(
        "--quote-symbol-ratio",
        type=float,
        default=None,
        help="Minimum required-symbol overlap ratio in a diagnostic quote (strict mode only).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        metavar="N",
        help="Print a throughput summary every N completed samples; 0 disables.",
    )
    parser.add_argument(
        "--verbose-per-sample",
        action="store_true",
        help="Print one line before each sample (very noisy; use for deep debugging).",
    )

    args = parser.parse_args()

    if args.topic_skip_prediction:
        os.environ["PHYSICSVERIFIER_TOPIC_SKIP_PREDICTION"] = "1"

    with open(args.input, "r", encoding="utf-8") as f:
        samples = json.load(f)

    n_samples = len(samples) if isinstance(samples, list) else 0
    print(
        f"[PhysicsVerifier] loaded {n_samples} samples from {args.input!r} | "
        f"progress every {max(0, int(args.progress_interval))} (0=off)",
        flush=True,
    )

    verifier = PhysicsRuleVerifier(
        rules_catalog_path=args.catalog,
        llm_model=args.model,
        enable_symbolic_check=not args.no_symbolic_check,
        unified_rules_path=args.unified_catalog,
        experience_code_manifest_path=args.experience_code_manifest,
        experience_code_module=args.experience_code_module,
        symbolic_topic_check_limit=args.symbolic_topic_check_limit,
        precision_mode=args.precision_mode,
        min_diagnostic_rule_score=args.min_diagnostic_rule_score,
        max_diagnostics_per_sample=args.max_per_sample,
        max_diagnostics_per_paragraph=args.max_per_paragraph,
        quote_required_symbol_ratio=args.quote_symbol_ratio,
        unified_rule_top_n=args.unified_rule_top_n,
    )

    raw_results = verifier.run_batch(
        samples,
        progress_interval=max(0, int(args.progress_interval)),
        verbose_per_sample=bool(args.verbose_per_sample),
    )

    # Main output: only final diagnostics (after symbolic suppression), without symbolic metadata.
    results = [_build_main_result(r) for r in (raw_results or []) if isinstance(r, dict)]

    # Symbolic audit output: only samples that had symbolic checks or produced symbolic outputs.
    symbolic_audit_all = [_build_symbolic_audit(r) for r in (raw_results or []) if isinstance(r, dict)]
    symbolic_audit = [
        a
        for a in symbolic_audit_all
        if (
            a.get("checked_diagnostics")
            or a.get("symbolic_checks")
            or a.get("experience_code_checks")
            or a.get("suppressed_diagnostics")
        )
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    sym_path = Path(args.symbolic_output)
    sym_path.parent.mkdir(parents=True, exist_ok=True)
    sym_path.write_text(json.dumps(symbolic_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.full_output:
        full_path = Path(args.full_output)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(json.dumps(raw_results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Done. Results saved to {out_path}")
    print(f"Done. Symbolic audit saved to {sym_path}")
    if args.full_output:
        print(f"Done. Full raw results saved to {args.full_output}")


if __name__ == "__main__":
    main()
