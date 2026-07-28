"""Replay the new symbolic-check + release-gate logic on cached verifier output.

This avoids re-running expensive LLM inference. It is mainly used for
ablation/debug: given a previous ``error_verifier_results.json``, re-apply the
current symbolic executor + release gate semantics to the same candidate
diagnostics and emit a new ``error_verifier_results.json`` that downstream
``evaluate_physics_eval_sets.py`` can score.

Usage:
    .venv/bin/python scripts/replay_symbolic_and_gate.py \
        --dataset path/to/error_eval_dataset.json \
        --prior-results path/to/error_verifier_results.json \
        --output path/to/replayed_results.json \
        --catalog catalogs/legacy/unified_rule_library_v2_llm_enhanced_20260504.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.physics_rule_verifier import PhysicsRuleVerifier  # noqa: E402


def _build_dataset_index(dataset_path: Path) -> Dict[str, Dict[str, Any]]:
    samples = json.loads(dataset_path.read_text(encoding="utf-8"))
    return {str(s.get("id")): s for s in samples}


def _restore_candidate_diagnostics(prior: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The saved file strips ``symbolic_*`` fields; the candidate diagnostics
    are still the raw outputs of the LLM-based semantic checker, so we feed
    them back into the verifier as if they had just been produced.
    """
    diags = prior.get("candidate_diagnostics") or prior.get("diagnostics") or []
    out: List[Dict[str, Any]] = []
    for d in diags:
        if isinstance(d, dict):
            cleaned = {k: v for k, v in d.items() if k not in {"symbolic_cross_checks", "symbolic_reconciliation", "release_gate", "rule_match"}}
            out.append(cleaned)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prior-results", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--catalog",
        required=True,
        help="Unified rules catalog used by the prior run.",
    )
    parser.add_argument(
        "--unified-retrieval-mode",
        choices=["lexical", "semantic"],
        default="lexical",
        help=(
            "Rule retrieval mode used during replay. Defaults to lexical so this offline "
            "symbolic/gate replay does not call the semantic-model API."
        ),
    )
    parser.add_argument("--precision-mode", default="strict")
    parser.add_argument("--max-per-sample", type=int, default=None)
    parser.add_argument("--max-per-paragraph", type=int, default=None)
    parser.add_argument("--quote-symbol-ratio", type=float, default=None)
    args = parser.parse_args()

    dataset_index = _build_dataset_index(Path(args.dataset))
    prior_records = json.loads(Path(args.prior_results).read_text(encoding="utf-8"))

    verifier = PhysicsRuleVerifier(
        unified_rules_path=args.catalog,
        unified_retrieval_mode=args.unified_retrieval_mode,
        enable_symbolic_check=False,
        precision_mode=args.precision_mode,
        max_diagnostics_per_sample=args.max_per_sample,
        max_diagnostics_per_paragraph=args.max_per_paragraph,
        quote_required_symbol_ratio=args.quote_symbol_ratio,
    )

    cached = {}

    def _patched_analyze(sample: Dict[str, Any]) -> Dict[str, Any]:
        sid = str(sample.get("id"))
        return {"diagnostics": list(cached.get(sid, []))}

    verifier.semantic_checker.analyze = _patched_analyze  # type: ignore[assignment]

    out_records: List[Dict[str, Any]] = []
    for prior in prior_records:
        sid = str(prior.get("id"))
        sample = dataset_index.get(sid)
        if sample is None:
            print(f"[WARN] no dataset sample for id={sid}; skipping")
            continue
        cached[sid] = _restore_candidate_diagnostics(prior)
        sample_input = dict(sample)
        if hasattr(verifier, "verify"):
            result = verifier.verify(sample_input)
        elif hasattr(verifier, "process"):
            result = verifier.process(sample_input)
        else:  # pragma: no cover - safety
            raise RuntimeError("Verifier missing verify/process entry point")
        # Mirror the run_verifier.py output shape
        cleaned_diags = []
        for d in result.get("diagnostics", []) or []:
            if isinstance(d, dict):
                ed = dict(d)
                ed.pop("symbolic_cross_checks", None)
                ed.pop("symbolic_reconciliation", None)
                cleaned_diags.append(ed)
        out_records.append(
            {
                "id": result.get("id"),
                "topic": result.get("topic"),
                "verifier": result.get("verifier"),
                "diagnostics": cleaned_diags,
                "score": result.get("score"),
            }
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(out_records)} replayed records to {args.output}")


if __name__ == "__main__":
    main()
