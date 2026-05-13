from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.rule_catalog_retrieval import (
    apply_topic_symbol_overlap_boost,
    build_signal_document_frequency,
    build_topic_candidates,
    build_unified_topic_retrieval_text,
    extract_prediction_symbol_set,
    norm_text,
    rule_topic_context,
    rule_sort_key,
    select_rules_with_topic_priority,
    score_rule_candidate,
    score_topic_candidate,
    topic_rule_leaves,
    topic_sort_key,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_retrieval_tuning_into_catalog(catalog: Dict[str, Any], tuning: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of `catalog` with `metadata.retrieval_tuning` merged with `tuning`."""
    out = copy.deepcopy(catalog)
    meta = out.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
        out["metadata"] = meta
    base = meta.get("retrieval_tuning")
    base_d = dict(base) if isinstance(base, dict) else {}
    base_d.update(tuning)
    meta["retrieval_tuning"] = base_d
    return out


def _norm_text(text: Any) -> str:
    return norm_text(text)


def _snippet(text: Any, limit: int = 240) -> str:
    value = _norm_text(text)
    return value[:limit]


def _p95(values: List[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return float(ordered[idx])


def retrieve_topics(
    topic_candidates: List[Dict[str, Any]],
    signal_df: Dict[str, Any],
    sample: Dict[str, Any],
    *,
    top_k: int,
    retrieval_tuning: Optional[Dict[str, Any]] = None,
    legacy_topic_text: bool = False,
    apply_symbol_boost: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    tuning = retrieval_tuning or {}
    if legacy_topic_text:
        text_for_topic = "\n".join(
            [
                _norm_text(sample.get("question") or ""),
                _norm_text(sample.get("context") or ""),
            ]
        )
    else:
        text_for_topic = build_unified_topic_retrieval_text(sample, tuning=tuning)
    scored = [
        score_topic_candidate(candidate, text_for_topic, signal_df=signal_df)
        for candidate in topic_candidates
    ]
    pred_syms = extract_prediction_symbol_set(str(sample.get("prediction") or ""))
    if apply_symbol_boost:
        apply_topic_symbol_overlap_boost(scored, pred_syms, tuning=tuning)
    scored.sort(key=topic_sort_key)
    selected = scored[:top_k]
    trace = [
        {
            "domain": item["domain"],
            "topic": item["topic"],
            "score": item["score"],
            "evidence": item["evidence"],
        }
        for item in scored[: min(len(scored), 12)]
    ]
    return selected, trace


def retrieve_rules(
    topic_matches: List[Dict[str, Any]],
    sample: Dict[str, Any],
    *,
    top_n: int,
    retrieval_tuning: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    text_for_rule = "\n".join(
        [
            _norm_text(sample.get("question") or ""),
            _norm_text(sample.get("context") or ""),
            _norm_text(sample.get("prediction") or ""),
        ]
    )
    scored: List[Dict[str, Any]] = []
    top1_score = float(topic_matches[0]["score"]) if topic_matches else 0.0
    top1_margin = top1_score - float(topic_matches[1]["score"]) if len(topic_matches) > 1 else top1_score
    top1_key = None
    if topic_matches:
        top1_key = (str(topic_matches[0]["domain"] or ""), str(topic_matches[0]["topic"] or ""))
    for topic_rank, topic_match in enumerate(topic_matches):
        topic_obj = topic_match.get("topic_obj") if isinstance(topic_match.get("topic_obj"), dict) else {}
        for rule in topic_rule_leaves(topic_obj):
            if not isinstance(rule, dict):
                continue
            payload = score_rule_candidate(rule, text_for_rule)
            topic_ctx = rule_topic_context(
                raw_score=float(payload["score"] or 0.0),
                topic_rank=topic_rank,
                topic_score=float(topic_match.get("score") or 0.0),
                top1_topic_score=top1_score,
                scope=str(payload.get("scope") or "domain"),
                rule_evidence=payload.get("evidence") or {},
                topic_evidence=topic_match.get("evidence") or {},
                retrieval_tuning=retrieval_tuning,
            )
            scored.append(
                {
                    "rule_id": payload["rule_id"],
                    "domain": topic_match["domain"],
                    "topic": topic_match["topic"],
                    "title": payload["title"],
                    "score": payload["score"],
                    "adjusted_score": topic_ctx["adjusted_score"],
                    "topic_gap": topic_ctx["topic_gap"],
                    "min_score": topic_ctx["min_score"],
                    "scope": payload["scope"],
                    "manual_override_reason": str((payload.get("evidence") or {}).get("manual_override_reason") or ""),
                    "evidence": payload["evidence"],
                }
            )

    scored.sort(key=rule_sort_key)
    selected = select_rules_with_topic_priority(
        [item for item in scored if item["score"] > 0],
        top_n=top_n,
        top1_key=top1_key,
        top1_margin=float(top1_margin or 0.0),
    )
    trace = scored[: min(len(scored), 20)]
    return selected, trace


def _effective_retrieval_tuning(
    catalog: Dict[str, Any],
    override: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    meta = catalog.get("metadata") if isinstance(catalog.get("metadata"), dict) else {}
    base = meta.get("retrieval_tuning") if isinstance(meta.get("retrieval_tuning"), dict) else {}
    out = dict(base)
    if override:
        out.update(override)
    return out


def tuning_objective(summary: Dict[str, Any]) -> float:
    """Lower is better for offline sweep (heuristic composite)."""
    return (
        float(summary.get("rules_outside_top1_topic_ratio") or 0.0)
        + 0.5 * float(summary.get("low_margin_cross_topic_ratio") or 0.0)
        + 0.25 * float(summary.get("samples_with_generic_signal_rules_ratio") or 0.0)
    )


def sweep_retrieval_tuning_grid(
    catalog: Dict[str, Any],
    samples: List[Dict[str, Any]],
    *,
    top_topics: int,
    top_rules: int,
    annotation_limit: int,
    legacy_topic_text: bool,
    apply_topic_symbol_boost: bool,
    retrieval_override: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Small grid over deltas + topic_include_prediction; returns (best_tuning, all_results)."""
    deltas = [-0.5, 0.0, 0.5]
    pred_flags = [True, False]
    best: Optional[Dict[str, Any]] = None
    best_obj = float("inf")
    results: List[Dict[str, Any]] = []
    for ms, bd, pred in product(deltas, deltas, pred_flags):
        tuning = dict(_effective_retrieval_tuning(catalog, retrieval_override))
        tuning["min_score_delta"] = ms
        tuning["bonus_delta"] = bd
        tuning["topic_include_prediction"] = pred
        analysis = analyze_matching(
            catalog,
            samples,
            top_topics=top_topics,
            top_rules=top_rules,
            annotation_limit=annotation_limit,
            retrieval_tuning=tuning,
            legacy_topic_text=legacy_topic_text,
            apply_topic_symbol_boost=apply_topic_symbol_boost,
        )
        obj = tuning_objective(analysis["summary"])
        results.append({"tuning": tuning, "objective": obj, "summary": analysis["summary"]})
        if obj < best_obj:
            best_obj = obj
            best = tuning
    assert best is not None
    return best, results


def analyze_matching(
    catalog: Dict[str, Any],
    samples: List[Dict[str, Any]],
    *,
    top_topics: int = 3,
    top_rules: int = 6,
    annotation_limit: int = 50,
    retrieval_tuning: Optional[Dict[str, Any]] = None,
    legacy_topic_text: bool = False,
    apply_topic_symbol_boost: bool = True,
) -> Dict[str, Any]:
    topic_candidates = build_topic_candidates(catalog)
    signal_df = build_signal_document_frequency(topic_candidates)
    effective_tuning = _effective_retrieval_tuning(catalog, retrieval_tuning)

    per_sample: List[Dict[str, Any]] = []
    annotation_samples: List[Dict[str, Any]] = []
    candidate_counts: List[int] = []
    positive_rule_trace_counts: List[int] = []
    topic_margins: List[float] = []
    saturated_rule_samples = 0
    outside_top1_topic_samples = 0
    meta_rule_samples = 0
    low_margin_samples = 0
    low_margin_cross_topic_samples = 0
    strong_top1_cross_topic_samples = 0
    manual_override_rule_samples = 0
    generic_signal_rule_samples = 0

    for sample in samples:
        topic_matches, topic_trace = retrieve_topics(
            topic_candidates,
            signal_df,
            sample,
            top_k=top_topics,
            retrieval_tuning=effective_tuning,
            legacy_topic_text=legacy_topic_text,
            apply_symbol_boost=apply_topic_symbol_boost,
        )
        rule_matches, rule_trace = retrieve_rules(
            topic_matches, sample, top_n=top_rules, retrieval_tuning=effective_tuning
        )
        positive_rule_trace_count = sum(1 for item in rule_trace if float(item.get("score") or 0.0) > 0)
        topic_top1_score = float(topic_matches[0]["score"]) if topic_matches else 0.0
        topic_top2_score = float(topic_matches[1]["score"]) if len(topic_matches) > 1 else 0.0
        topic_margin = round(topic_top1_score - topic_top2_score, 4)
        rule_topics = {
            (str(item.get("domain") or ""), str(item.get("topic") or ""))
            for item in rule_matches
        }
        top1_key = None
        if topic_matches:
            top1_key = (str(topic_matches[0]["domain"] or ""), str(topic_matches[0]["topic"] or ""))
        rules_outside_top1_topic = bool(top1_key and any(key != top1_key for key in rule_topics))
        meta_rule_count = sum(1 for item in rule_matches if str(item.get("scope") or "") == "meta")
        manual_override_rule_count = sum(1 for item in rule_matches if str(item.get("manual_override_reason") or ""))
        generic_signal_rule_count = sum(
            1 for item in rule_matches if bool((item.get("evidence") or {}).get("generic_signal_only"))
        )
        strong_top1_cross_topic = bool(topic_margin >= 3.0 and rules_outside_top1_topic)
        low_margin_topic = bool(topic_margin <= 0.5)

        record = {
            "id": sample.get("id"),
            "retrieved_topics": [
                {
                    "domain": item["domain"],
                    "topic": item["topic"],
                    "score": item["score"],
                    "evidence": item["evidence"],
                }
                for item in topic_matches
            ],
            "retrieved_rules": rule_matches,
            "topic_score_trace": topic_trace,
            "rule_score_trace": rule_trace,
            "candidate_rule_count": len(rule_matches),
            "positive_rule_trace_count": positive_rule_trace_count,
            "topic_top1_score": topic_top1_score,
            "topic_top2_score": topic_top2_score,
            "topic_score_margin": topic_margin,
            "rule_trace_overflow_count": max(0, positive_rule_trace_count - top_rules),
            "meta_rule_count": meta_rule_count,
            "manual_override_rule_count": manual_override_rule_count,
            "generic_signal_rule_count": generic_signal_rule_count,
            "rules_outside_top1_topic": rules_outside_top1_topic,
            "strong_top1_cross_topic": strong_top1_cross_topic,
            "low_margin_topic": low_margin_topic,
            "question_snippet": _snippet(sample.get("question")),
            "prediction_snippet": _snippet(sample.get("prediction")),
        }
        per_sample.append(record)
        candidate_counts.append(len(rule_matches))
        positive_rule_trace_counts.append(positive_rule_trace_count)
        topic_margins.append(topic_margin)
        if len(rule_matches) >= top_rules:
            saturated_rule_samples += 1
        if rules_outside_top1_topic:
            outside_top1_topic_samples += 1
        if meta_rule_count > 0:
            meta_rule_samples += 1
        if low_margin_topic:
            low_margin_samples += 1
        if low_margin_topic and rules_outside_top1_topic:
            low_margin_cross_topic_samples += 1
        if strong_top1_cross_topic:
            strong_top1_cross_topic_samples += 1
        if manual_override_rule_count > 0:
            manual_override_rule_samples += 1
        if generic_signal_rule_count > 0:
            generic_signal_rule_samples += 1

        if len(annotation_samples) < annotation_limit:
            annotation_samples.append(
                {
                    "id": sample.get("id"),
                    "question": sample.get("question"),
                    "prediction": sample.get("prediction"),
                    "answer": sample.get("answer"),
                    "retrieved_topics": record["retrieved_topics"],
                    "retrieved_rules": record["retrieved_rules"],
                    "candidate_rule_count": len(rule_matches),
                    "positive_rule_trace_count": positive_rule_trace_count,
                    "topic_score_margin": topic_margin,
                    "meta_rule_count": meta_rule_count,
                    "manual_override_rule_count": manual_override_rule_count,
                    "generic_signal_rule_count": generic_signal_rule_count,
                    "rules_outside_top1_topic": rules_outside_top1_topic,
                    "strong_top1_cross_topic": strong_top1_cross_topic,
                    "low_margin_topic": low_margin_topic,
                    "topic_match": "",
                    "rule_match": "",
                    "retrieval_notes": "",
                }
            )

    average = round(sum(candidate_counts) / len(candidate_counts), 4) if candidate_counts else 0.0
    summary = {
        "total_samples": len(samples),
        "top_topics": top_topics,
        "top_rules": top_rules,
        "annotation_limit": annotation_limit,
        "legacy_topic_text": legacy_topic_text,
        "apply_topic_symbol_boost": apply_topic_symbol_boost,
        "retrieval_tuning_effective": effective_tuning,
        "average_candidate_rule_count": average,
        "p95_candidate_rule_count": _p95(candidate_counts),
        "max_candidate_rule_count": max(candidate_counts) if candidate_counts else 0,
        "min_candidate_rule_count": min(candidate_counts) if candidate_counts else 0,
        "average_positive_rule_trace_count": round(sum(positive_rule_trace_counts) / len(positive_rule_trace_counts), 4)
        if positive_rule_trace_counts
        else 0.0,
        "average_topic_score_margin": round(sum(topic_margins) / len(topic_margins), 4) if topic_margins else 0.0,
        "rule_topk_saturation_count": saturated_rule_samples,
        "rule_topk_saturation_ratio": round(saturated_rule_samples / len(samples), 4) if samples else 0.0,
        "rules_outside_top1_topic_count": outside_top1_topic_samples,
        "rules_outside_top1_topic_ratio": round(outside_top1_topic_samples / len(samples), 4) if samples else 0.0,
        "samples_with_meta_rules_count": meta_rule_samples,
        "samples_with_meta_rules_ratio": round(meta_rule_samples / len(samples), 4) if samples else 0.0,
        "low_margin_topic_count": low_margin_samples,
        "low_margin_topic_ratio": round(low_margin_samples / len(samples), 4) if samples else 0.0,
        "low_margin_cross_topic_count": low_margin_cross_topic_samples,
        "low_margin_cross_topic_ratio": round(low_margin_cross_topic_samples / len(samples), 4) if samples else 0.0,
        "strong_top1_cross_topic_count": strong_top1_cross_topic_samples,
        "strong_top1_cross_topic_ratio": round(strong_top1_cross_topic_samples / len(samples), 4) if samples else 0.0,
        "samples_with_manual_override_rules_count": manual_override_rule_samples,
        "samples_with_manual_override_rules_ratio": round(manual_override_rule_samples / len(samples), 4)
        if samples
        else 0.0,
        "samples_with_generic_signal_rules_count": generic_signal_rule_samples,
        "samples_with_generic_signal_rules_ratio": round(generic_signal_rule_samples / len(samples), 4)
        if samples
        else 0.0,
    }
    summary["tuning_objective"] = tuning_objective(summary)
    return {
        "summary": summary,
        "per_sample": per_sample,
        "annotation_samples": annotation_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze offline matching quality for unified_rules v2.")
    parser.add_argument("--catalog", default="catalogs/unified_rule_library.json")
    parser.add_argument("--input", default="data/evaluation_sample_300.json")
    parser.add_argument("--outdir", default="results/unified_matching_analysis")
    parser.add_argument("--top-topics", type=int, default=3)
    parser.add_argument("--top-rules", type=int, default=6)
    parser.add_argument("--annotation-limit", type=int, default=50)
    parser.add_argument(
        "--legacy-topic-text",
        action="store_true",
        help="Use only question+context for topic scoring (A/B vs default extended text).",
    )
    parser.add_argument(
        "--no-topic-symbol-boost",
        action="store_true",
        help="Disable prediction-symbol overlap boost on topic scores.",
    )
    parser.add_argument(
        "--tuning-json",
        type=str,
        default="",
        help="Optional JSON file of retrieval_tuning overrides (merged on top of catalog metadata).",
    )
    parser.add_argument(
        "--sweep-tuning",
        action="store_true",
        help="Grid-search min_score_delta, bonus_delta, topic_include_prediction; writes sweep_results + best_tuning.json",
    )
    parser.add_argument(
        "--export-catalog-with-tuning",
        type=str,
        default="",
        help="If set with --sweep-tuning, write catalog copy with best tuning embedded to this path.",
    )
    parser.add_argument(
        "--ab-compare-topic-text",
        action="store_true",
        help="Run legacy vs extended topic text and write topic_text_ab.json with both summaries.",
    )
    args = parser.parse_args()

    catalog = _load_json(Path(args.catalog))
    samples = _load_json(Path(args.input))
    if not isinstance(samples, list):
        raise SystemExit("Input samples must be a JSON list.")

    tuning_override: Optional[Dict[str, Any]] = None
    if str(args.tuning_json or "").strip():
        p = Path(args.tuning_json)
        if not p.exists():
            raise SystemExit(f"--tuning-json not found: {p}")
        raw = _load_json(p)
        if not isinstance(raw, dict):
            raise SystemExit("--tuning-json must be a JSON object")
        tuning_override = raw

    outdir = Path(args.outdir)
    top_topics = max(1, int(args.top_topics))
    top_rules = max(1, int(args.top_rules))
    ann = max(1, int(args.annotation_limit))
    use_legacy = bool(args.legacy_topic_text)
    use_symbol_boost = not bool(args.no_topic_symbol_boost)

    if args.sweep_tuning:
        best, grid = sweep_retrieval_tuning_grid(
            catalog,
            samples,
            top_topics=top_topics,
            top_rules=top_rules,
            annotation_limit=ann,
            legacy_topic_text=use_legacy,
            apply_topic_symbol_boost=use_symbol_boost,
            retrieval_override=tuning_override,
        )
        outdir.mkdir(parents=True, exist_ok=True)
        _dump_json(outdir / "sweep_results.json", {"best_tuning": best, "grid": grid})
        _dump_json(outdir / "best_tuning.json", best)
        if str(args.export_catalog_with_tuning or "").strip():
            merged = merge_retrieval_tuning_into_catalog(catalog, best)
            _dump_json(Path(args.export_catalog_with_tuning), merged)
        print(f"[ok] sweep: best_tuning -> {outdir / 'best_tuning.json'}")

    if args.ab_compare_topic_text:
        ext = analyze_matching(
            catalog,
            samples,
            top_topics=top_topics,
            top_rules=top_rules,
            annotation_limit=ann,
            retrieval_tuning=tuning_override,
            legacy_topic_text=False,
            apply_topic_symbol_boost=use_symbol_boost,
        )
        leg = analyze_matching(
            catalog,
            samples,
            top_topics=top_topics,
            top_rules=top_rules,
            annotation_limit=ann,
            retrieval_tuning=tuning_override,
            legacy_topic_text=True,
            apply_topic_symbol_boost=use_symbol_boost,
        )
        outdir.mkdir(parents=True, exist_ok=True)
        _dump_json(
            outdir / "topic_text_ab.json",
            {
                "extended_topic_text_summary": ext["summary"],
                "legacy_topic_text_summary": leg["summary"],
            },
        )
        print(f"[ok] wrote: {outdir / 'topic_text_ab.json'}")

    analysis = analyze_matching(
        catalog,
        samples,
        top_topics=top_topics,
        top_rules=top_rules,
        annotation_limit=ann,
        retrieval_tuning=tuning_override,
        legacy_topic_text=use_legacy,
        apply_topic_symbol_boost=use_symbol_boost,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    _dump_json(outdir / "summary.json", analysis["summary"])
    _dump_json(outdir / "per_sample.json", analysis["per_sample"])
    _dump_json(outdir / "annotation_samples.json", analysis["annotation_samples"])

    report_lines = [
        "# Unified Rules V2 Matching Analysis",
        "",
        f"- samples: {analysis['summary']['total_samples']}",
        f"- average candidate rule count: {analysis['summary']['average_candidate_rule_count']}",
        f"- p95 candidate rule count: {analysis['summary']['p95_candidate_rule_count']}",
        f"- max candidate rule count: {analysis['summary']['max_candidate_rule_count']}",
        f"- min candidate rule count: {analysis['summary']['min_candidate_rule_count']}",
        f"- average positive rule trace count: {analysis['summary']['average_positive_rule_trace_count']}",
        f"- average top1-top2 topic margin: {analysis['summary']['average_topic_score_margin']}",
        f"- rule top-k saturation ratio: {analysis['summary']['rule_topk_saturation_ratio']}",
        f"- rules outside top1 topic ratio: {analysis['summary']['rules_outside_top1_topic_ratio']}",
        f"- samples with meta rules ratio: {analysis['summary']['samples_with_meta_rules_ratio']}",
        f"- low-margin topic ratio: {analysis['summary']['low_margin_topic_ratio']}",
        f"- low-margin cross-topic ratio: {analysis['summary']['low_margin_cross_topic_ratio']}",
        f"- strong-top1 cross-topic ratio: {analysis['summary']['strong_top1_cross_topic_ratio']}",
        f"- samples with manual override rules ratio: {analysis['summary']['samples_with_manual_override_rules_ratio']}",
        f"- samples with generic-signal rules ratio: {analysis['summary']['samples_with_generic_signal_rules_ratio']}",
        f"- top topics kept: {analysis['summary']['top_topics']}",
        f"- top rules kept: {analysis['summary']['top_rules']}",
    ]
    (outdir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"[ok] wrote: {outdir / 'summary.json'}")
    print(f"[ok] wrote: {outdir / 'per_sample.json'}")
    print(f"[ok] wrote: {outdir / 'annotation_samples.json'}")
    print(f"[ok] wrote: {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
