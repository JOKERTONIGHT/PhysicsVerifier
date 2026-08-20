from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.physics_rule_verifier import PhysicsRuleVerifier
from core.rule_catalog_retrieval import (
    apply_topic_symbol_overlap_boost,
    build_unified_topic_retrieval_text,
    extract_prediction_symbol_set,
    score_rule_candidate,
    score_topic_candidate,
    topic_sort_key,
)


def _load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _topic_key(item: Dict[str, Any]) -> str:
    return f"{str(item.get('domain') or 'Unknown')}::{str(item.get('name') or item.get('topic') or 'Unknown')}"


def _topic_preview(items: Sequence[Dict[str, Any]], limit: int = 3) -> List[Dict[str, Any]]:
    return [
        {
            "topic": _topic_key(item),
            "score": round(float(item.get("score") or 0.0), 4),
            "evidence": item.get("evidence") or {},
        }
        for item in items[: max(0, int(limit))]
    ]


def _score_topics(
    verifier: PhysicsRuleVerifier,
    text: str,
    *,
    prediction_symbols: Iterable[str] = (),
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    scored: List[Dict[str, Any]] = []
    for candidate in verifier._unified_v2_topic_candidates:
        payload = score_topic_candidate(candidate, text, signal_df=verifier._unified_v2_signal_df)
        scored.append(
            {
                "domain": payload["domain"],
                "name": payload["topic"],
                "score": payload["score"],
                "evidence": payload["evidence"],
                "topic": payload["topic_obj"],
            }
        )
    pred_syms = {str(x) for x in prediction_symbols if str(x)}
    if pred_syms:
        apply_topic_symbol_overlap_boost(scored, pred_syms, tuning=verifier._retrieval_tuning)
    scored.sort(
        key=lambda item: topic_sort_key(
            {"score": item["score"], "domain": item["domain"], "topic": item["name"]}
        )
    )
    return scored[: max(1, int(top_k))]


def _topic_text_surfaces(sample: Dict[str, Any], tuning: Dict[str, Any]) -> Dict[str, str]:
    """Build production-equivalent topic text with one source changed at a time."""
    question = str(sample.get("question") or "")
    context = str(sample.get("context") or "")
    solution = str(sample.get("prediction") or "")
    return {
        "current": build_unified_topic_retrieval_text(sample, tuning=tuning),
        "problem_only": build_unified_topic_retrieval_text(
            {"question": question, "context": context, "prediction": ""},
            tuning=tuning,
        ),
        "solution_only": build_unified_topic_retrieval_text(
            {"question": "", "context": "", "prediction": solution},
            tuning=tuning,
        ),
    }


def _hit_set(payload: Dict[str, Any], field: str) -> set[str]:
    evidence = payload.get("evidence") if isinstance(payload.get("evidence"), dict) else {}
    return {str(x) for x in (evidence.get(field) or []) if str(x).strip()}


def _surface_mismatches(
    mixed_payload: Dict[str, Any],
    problem_payload: Dict[str, Any],
    solution_payload: Dict[str, Any],
) -> Dict[str, List[str]]:
    """Find signals whose current mixed-text hit comes only from the wrong source surface.

    These rows are candidates for manual review, not automatic correctness labels: some
    legacy free-text fields mix applicability and answer-behaviour semantics.
    """
    mapping = {
        "precondition_from_solution_only": "precondition_hits",
        "violation_from_problem_only": "violation_signature_hits",
        "evidence_requirement_from_problem_only": "evidence_requirement_hits",
    }
    out: Dict[str, List[str]] = {}
    for label, field in mapping.items():
        mixed = _hit_set(mixed_payload, field)
        problem = _hit_set(problem_payload, field)
        solution = _hit_set(solution_payload, field)
        if label == "precondition_from_solution_only":
            values = sorted(mixed.intersection(solution).difference(problem))
        else:
            values = sorted(mixed.intersection(problem).difference(solution))
        if values:
            out[label] = values
    return out


def _clip(text: Any, limit: int = 360) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: max(0, limit - 3)] + "..."


def _ratio(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def analyze(
    *,
    catalog_path: str,
    input_path: str,
    limit: int,
    top_k: int,
    rule_top_n: int,
    top_cases: int,
    skip_rule_audit: bool = False,
    sample_ids: Sequence[str] = (),
) -> Dict[str, Any]:
    samples = _load_json(input_path)
    if not isinstance(samples, list):
        raise ValueError("Input dataset must be a JSON array")
    wanted_ids = {str(x).strip() for x in sample_ids if str(x).strip()}
    if wanted_ids:
        samples = [
            sample
            for sample in samples
            if isinstance(sample, dict) and str(sample.get("id") or "") in wanted_ids
        ]
    if limit > 0:
        samples = samples[:limit]

    verifier = PhysicsRuleVerifier(
        unified_rules_path=catalog_path,
        llm_model=None,
        enable_symbolic_check=False,
        unified_rule_top_n=rule_top_n,
        unified_retrieval_mode="lexical",
    )
    if not verifier._unified_v2_mode:
        raise ValueError("Catalog is not recognized as unified_rules_v2")

    counts: Counter[str] = Counter()
    topic_cases: List[Dict[str, Any]] = []
    rule_cases: List[Dict[str, Any]] = []
    timings = Counter()
    wall_start = time.perf_counter()

    for sample in samples:
        if not isinstance(sample, dict):
            continue
        counts["samples"] += 1
        sid = str(sample.get("id") or "")
        question = str(sample.get("question") or "")
        context = str(sample.get("context") or "")
        solution = str(sample.get("prediction") or "")
        problem = "\n".join(x for x in (question, context) if x)
        mixed = "\n".join(x for x in (problem, solution) if x)
        # Mirror the production topic input exactly, including prediction
        # truncation. The counterfactuals change only the information source.
        topic_surfaces = _topic_text_surfaces(sample, verifier._retrieval_tuning)
        pred_syms = extract_prediction_symbol_set(solution)

        t0 = time.perf_counter()
        current_topics = _score_topics(
            verifier,
            topic_surfaces["current"],
            prediction_symbols=pred_syms,
            top_k=top_k,
        )
        timings["current_topic_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        problem_topics = _score_topics(verifier, topic_surfaces["problem_only"], top_k=top_k)
        timings["problem_topic_seconds"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        solution_topics = _score_topics(
            verifier,
            topic_surfaces["solution_only"],
            prediction_symbols=pred_syms,
            top_k=top_k,
        )
        timings["solution_topic_seconds"] += time.perf_counter() - t0

        current_top1 = _topic_key(current_topics[0]) if current_topics else ""
        problem_top1 = _topic_key(problem_topics[0]) if problem_topics else ""
        solution_top1 = _topic_key(solution_topics[0]) if solution_topics else ""
        current_top3 = {_topic_key(x) for x in current_topics}
        problem_top3 = {_topic_key(x) for x in problem_topics}

        if current_top1 != problem_top1:
            counts["topic_top1_changed"] += 1
        if current_top3 != problem_top3:
            counts["topic_topk_set_changed"] += 1
        problem_dropped = bool(problem_top1 and problem_top1 not in current_top3)
        if problem_dropped:
            counts["problem_top1_dropped_from_current_topk"] += 1
        prediction_dominant = bool(
            current_top1 and current_top1 == solution_top1 and current_top1 != problem_top1
        )
        if prediction_dominant:
            counts["prediction_dominant_topic_shift"] += 1

        if current_top1 != problem_top1 or wanted_ids:
            topic_cases.append(
                {
                    "id": sid,
                    "top1_changed": current_top1 != problem_top1,
                    "priority": int(problem_dropped) * 2 + int(prediction_dominant),
                    "prediction_dominant": prediction_dominant,
                    "problem_top1_dropped": problem_dropped,
                    "question": _clip(question),
                    "prediction": _clip(solution),
                    "current_topics": _topic_preview(current_topics),
                    "problem_only_topics": _topic_preview(problem_topics),
                    "solution_only_topics": _topic_preview(solution_topics),
                }
            )

        selected_rules: List[Dict[str, Any]] = []
        if not skip_rule_audit:
            t0 = time.perf_counter()
            selected_rules = verifier._retrieve_unified_v2_rules(
                current_topics,
                sample,
                top_n=rule_top_n,
            )
            timings["rule_retrieval_seconds"] += time.perf_counter() - t0
        counts["selected_rules"] += len(selected_rules)

        for item in selected_rules:
            rule = item.get("rule") if isinstance(item.get("rule"), dict) else {}
            mixed_payload = score_rule_candidate(rule, mixed)
            problem_payload = score_rule_candidate(rule, problem)
            solution_payload = score_rule_candidate(rule, solution)
            problem_score = float(problem_payload.get("score") or 0.0)
            solution_score = float(solution_payload.get("score") or 0.0)
            if problem_score <= 0.0 and solution_score > 0.0:
                counts["selected_rules_solution_only_score"] += 1

            mismatches = _surface_mismatches(mixed_payload, problem_payload, solution_payload)
            if not mismatches:
                continue
            counts["selected_rules_with_source_mismatch"] += 1
            for label in mismatches:
                counts[label] += 1

            gate = item.get("publish_gate") if isinstance(item.get("publish_gate"), dict) else {}
            if gate.get("publishable"):
                counts["publishable_rules_with_source_mismatch"] += 1
                if "precondition_from_solution_only" in mismatches:
                    counts["publishable_rules_with_solution_only_precondition"] += 1

            rule_cases.append(
                {
                    "id": sid,
                    "rule_id": str(rule.get("id") or rule.get("rule_id") or ""),
                    "title": str(rule.get("title") or ""),
                    "domain": str(item.get("domain") or "Unknown"),
                    "topic": str(item.get("topic_name") or "Unknown"),
                    "metadata_origin": "reference_restored" if rule.get("source_rule_ids") else "deterministic_backfill",
                    "selected_score": round(float(item.get("score") or 0.0), 4),
                    "problem_score": round(problem_score, 4),
                    "solution_score": round(solution_score, 4),
                    "publishable": bool(gate.get("publishable")),
                    "gate_reasons": list(gate.get("reasons") or []),
                    "source_mismatches": mismatches,
                    "question": _clip(question),
                    "prediction": _clip(solution),
                }
            )

    samples_n = int(counts["samples"])
    selected_n = int(counts["selected_rules"])
    topic_cases.sort(
        key=lambda x: (-int(x.get("priority") or 0), str(x.get("id") or ""))
    )
    rule_cases.sort(
        key=lambda x: (
            -int(bool(x.get("publishable"))),
            -len(x.get("source_mismatches") or {}),
            -float(x.get("selected_score") or 0.0),
            str(x.get("id") or ""),
        )
    )
    wall_seconds = time.perf_counter() - wall_start

    summary = {
        "samples": samples_n,
        "topic_top1_changed": int(counts["topic_top1_changed"]),
        "topic_top1_changed_rate": _ratio(counts["topic_top1_changed"], samples_n),
        "topic_topk_set_changed": int(counts["topic_topk_set_changed"]),
        "topic_topk_set_changed_rate": _ratio(counts["topic_topk_set_changed"], samples_n),
        "prediction_dominant_topic_shift": int(counts["prediction_dominant_topic_shift"]),
        "prediction_dominant_topic_shift_rate": _ratio(counts["prediction_dominant_topic_shift"], samples_n),
        "problem_top1_dropped_from_current_topk": int(counts["problem_top1_dropped_from_current_topk"]),
        "problem_top1_dropped_from_current_topk_rate": _ratio(
            counts["problem_top1_dropped_from_current_topk"], samples_n
        ),
        "selected_rules": selected_n,
        "selected_rules_solution_only_score": int(counts["selected_rules_solution_only_score"]),
        "selected_rules_solution_only_score_rate": _ratio(
            counts["selected_rules_solution_only_score"], selected_n
        ),
        "selected_rules_with_source_mismatch": int(counts["selected_rules_with_source_mismatch"]),
        "selected_rules_with_source_mismatch_rate": _ratio(
            counts["selected_rules_with_source_mismatch"], selected_n
        ),
        "publishable_rules_with_source_mismatch": int(counts["publishable_rules_with_source_mismatch"]),
        "publishable_rules_with_solution_only_precondition": int(
            counts["publishable_rules_with_solution_only_precondition"]
        ),
        "source_mismatch_counts": {
            key: int(counts[key])
            for key in (
                "precondition_from_solution_only",
                "violation_from_problem_only",
                "evidence_requirement_from_problem_only",
            )
        },
        "wall_seconds": round(wall_seconds, 4),
        "samples_per_second": round(_ratio(samples_n, wall_seconds), 4),
        "timing_ms_per_sample": {
            "current_topic": round(1000.0 * _ratio(timings["current_topic_seconds"], samples_n), 4),
            "problem_only_topic": round(1000.0 * _ratio(timings["problem_topic_seconds"], samples_n), 4),
            "solution_only_topic": round(1000.0 * _ratio(timings["solution_topic_seconds"], samples_n), 4),
            "current_rule_retrieval": round(1000.0 * _ratio(timings["rule_retrieval_seconds"], samples_n), 4),
        },
    }

    return {
        "experiment": "verifier_context_source_audit.v1",
        "catalog": catalog_path,
        "input": input_path,
        "configuration": {
            "limit": limit,
            "topic_top_k": top_k,
            "rule_top_n": rule_top_n,
            "skip_rule_audit": bool(skip_rule_audit),
            "sample_ids": sorted(wanted_ids),
            "api_calls": 0,
        },
        "interpretation_guardrail": (
            "Topic shifts and source mismatches are screening signals, not automatic correctness labels. "
            "They require manual physics review because legacy metadata fields are free text."
        ),
        "summary": summary,
        "topic_shift_cases": topic_cases[: max(0, int(top_cases))],
        "rule_source_mismatch_cases": rule_cases[: max(0, int(top_cases))],
    }


def _pct(value: Any) -> str:
    return f"{100.0 * float(value or 0.0):.1f}%"


def render_markdown(report: Dict[str, Any], *, case_limit: int = 5) -> str:
    s = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    lines = [
        "# 题目背景接入：典型案例与无 API 离线实验",
        "",
        "## 实验说明",
        "",
        f"- 输入：`{report.get('input')}`",
        f"- 规则库：`{report.get('catalog')}`",
        f"- 样本数：{s.get('samples', 0)}",
        "- 本实验不调用 API。题目变更和字段来源错位只用于筛查，不能自动判定物理正误。",
        "",
        "## 核心结果",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 加入 prediction 后 top-1 topic 发生变化 | {s.get('topic_top1_changed', 0)} / {s.get('samples', 0)} ({_pct(s.get('topic_top1_changed_rate'))}) |",
        f"| 当前 top-1 与 solution-only 相同、但与 problem-only 不同 | {s.get('prediction_dominant_topic_shift', 0)} ({_pct(s.get('prediction_dominant_topic_shift_rate'))}) |",
        f"| problem-only top-1 掉出当前 top-K | {s.get('problem_top1_dropped_from_current_topk', 0)} ({_pct(s.get('problem_top1_dropped_from_current_topk_rate'))}) |",
        f"| 入选规则完全由 solution 侧获得词面分数 | {s.get('selected_rules_solution_only_score', 0)} / {s.get('selected_rules', 0)} ({_pct(s.get('selected_rules_solution_only_score_rate'))}) |",
        f"| 入选规则存在字段来源错位候选 | {s.get('selected_rules_with_source_mismatch', 0)} / {s.get('selected_rules', 0)} ({_pct(s.get('selected_rules_with_source_mismatch_rate'))}) |",
        f"| 吞吐 | {s.get('samples_per_second', 0)} samples/s |",
        "",
        "## Topic 变化案例（待人工复核）",
        "",
    ]
    for case in (report.get("topic_shift_cases") or [])[: max(0, int(case_limit))]:
        current = (case.get("current_topics") or [{}])[0].get("topic", "")
        problem = (case.get("problem_only_topics") or [{}])[0].get("topic", "")
        solution = (case.get("solution_only_topics") or [{}])[0].get("topic", "")
        lines.extend(
            [
                f"### {case.get('id')}",
                "",
                f"- current / problem-only / solution-only：`{current}` / `{problem}` / `{solution}`",
                f"- problem-only top-1 是否掉出当前 top-K：`{bool(case.get('problem_top1_dropped'))}`",
                f"- 题目：{case.get('question', '')}",
                f"- 解答：{case.get('prediction', '')}",
                "",
            ]
        )

    lines.extend(["## 规则字段来源错位案例（待人工复核）", ""])
    for case in (report.get("rule_source_mismatch_cases") or [])[: max(0, int(case_limit))]:
        lines.extend(
            [
                f"### {case.get('id')} · {case.get('rule_id')}",
                "",
                f"- 规则：{case.get('title', '')}（{case.get('domain')} / {case.get('topic')}）",
                f"- metadata 来源：`{case.get('metadata_origin')}`；publishable：`{case.get('publishable')}`",
                f"- mixed/problem/solution 分数：`{case.get('selected_score')}` / `{case.get('problem_score')}` / `{case.get('solution_score')}`",
                f"- 来源错位信号：`{json.dumps(case.get('source_mismatches') or {}, ensure_ascii=False)}`",
                f"- 题目：{case.get('question', '')}",
                f"- 解答：{case.get('prediction', '')}",
                "",
            ]
        )

    lines.extend(
        [
            "## 结论边界",
            "",
            "- 本实验能证明当前检索和门控对题设、学生答案的信息来源缺少隔离。",
            "- 本实验不能直接证明 problem-only 的 topic 一定正确；典型案例仍需人工物理复核。",
            "- 最终 recall/FP 效果必须等待新的 baseline/full verifier 输出后评估。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit problem/solution source mixing in verifier retrieval without API calls.")
    parser.add_argument("--catalog", default="catalogs/rules_unified_3000_runtime_backfilled.json")
    parser.add_argument("--input", default="data/evaluation_sample_300.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", default="")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--topic-top-k", type=int, default=3)
    parser.add_argument("--rule-top-n", type=int, default=6)
    parser.add_argument("--top-cases", type=int, default=20)
    parser.add_argument("--markdown-case-limit", type=int, default=5)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help="Analyze only this sample id; repeat the option for multiple ids.",
    )
    parser.add_argument(
        "--skip-rule-audit",
        action="store_true",
        help="Run only the faster topic-source experiment; omit per-rule source rescoring.",
    )
    args = parser.parse_args()

    report = analyze(
        catalog_path=args.catalog,
        input_path=args.input,
        limit=max(0, int(args.limit)),
        top_k=max(1, int(args.topic_top_k)),
        rule_top_n=max(1, int(args.rule_top_n)),
        top_cases=max(0, int(args.top_cases)),
        skip_rule_audit=bool(args.skip_rule_audit),
        sample_ids=args.sample_id,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_output:
        md = Path(args.markdown_output)
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(
            render_markdown(report, case_limit=max(0, int(args.markdown_case_limit))) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report["summary"], ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
