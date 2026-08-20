#!/usr/bin/env python3
"""Extract and classify failure cases from scale-curve eval results."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _tokenize(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(t) >= 3}


def _overlap(a: str, b: str) -> int:
    return len(_tokenize(a) & _tokenize(b))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finding_text(d: Dict[str, Any]) -> str:
    msg = str(d.get("message") or "").strip()
    ev = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
    quote = str(ev.get("quote") or "").strip()
    return f"{msg} {quote}".strip()


def _finding_para(d: Dict[str, Any]) -> int:
    ev = d.get("evidence") if isinstance(d.get("evidence"), dict) else {}
    loc = ev.get("location") if isinstance(ev.get("location"), dict) else {}
    return int(loc.get("paragraph_index") or -1)


def _gt_para(g: Dict[str, Any]) -> int:
    return int(g.get("paragraph_index") or -1)


def _classify_unmatched_gt(
    gt: Dict[str, Any],
    findings: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    gt_text = str(gt.get("error_text") or "")
    gt_para = _gt_para(gt)
    if not findings:
        return "no_detection", {"detail": "sample has no published diagnostics"}

    best_overlap = 0
    best_pred: Optional[Dict[str, Any]] = None
    same_para_preds = 0
    for f in findings:
        ft = _finding_text(f)
        ov = _overlap(gt_text, ft)
        if ov > best_overlap:
            best_overlap = ov
            best_pred = f
        if _finding_para(f) == gt_para and gt_para >= 1:
            same_para_preds += 1

    meta = {
        "best_token_overlap": best_overlap,
        "same_paragraph_preds": same_para_preds,
        "gt_paragraph_index": gt_para,
    }
    if best_pred:
        meta["best_pred_rule"] = str(best_pred.get("rule") or "")
        meta["best_pred_preview"] = _finding_text(best_pred)[:180]

    if best_overlap >= 4 and same_para_preds > 0:
        return "location_failure", meta
    if best_overlap >= 3:
        return "semantic_near_miss", meta
    if same_para_preds > 0 and best_overlap >= 1:
        return "rule_too_broad", meta
    return "semantic_gap", meta


def _classify_fp_pred(
    pred: Dict[str, Any],
    gt_items: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    pt = _finding_text(pred)
    p_para = _finding_para(pred)
    rule = str(pred.get("rule") or "")

    best_ov = 0
    best_gt: Optional[Dict[str, Any]] = None
    same_para_gt = 0
    for g in gt_items:
        gt_text = str(g.get("error_text") or "")
        ov = _overlap(pt, gt_text)
        if ov > best_ov:
            best_ov = ov
            best_gt = g
        if _gt_para(g) == p_para and p_para >= 1:
            same_para_gt += 1

    meta = {
        "rule": rule,
        "token_overlap_best_gt": best_ov,
        "same_paragraph_gt": same_para_gt,
        "pred_preview": pt[:180],
    }
    if best_gt:
        meta["nearest_gt_preview"] = str(best_gt.get("error_text") or "")[:180]

    if best_ov >= 4 and same_para_gt > 0:
        return "location_failure", meta
    if best_ov >= 3:
        return "semantic_near_miss", meta
    if rule.startswith("norm_") and best_ov <= 1:
        return "rule_too_broad", meta
    if same_para_gt > 0:
        return "rule_too_broad", meta
    return "irrelevant_trigger", meta


def _sample_primary_cause(
    detail: Dict[str, Any],
    gt_items: List[Dict[str, Any]],
    findings: List[Dict[str, Any]],
) -> str:
    gt_n = int(detail.get("gt_error_count") or 0)
    pred_n = int(detail.get("pred_finding_count") or 0)
    matched = int(detail.get("matched_error_count") or 0)
    fp_n = int(detail.get("unmatched_pred_locatable_count") or 0)

    if pred_n == 0 and gt_n > 0:
        return "no_detection"
    if matched == 0 and gt_n >= 5 and pred_n <= 2:
        return "gt_granularity_mismatch"
    if matched == 0 and pred_n > 0:
        gt_causes = [_classify_unmatched_gt(g, findings)[0] for g in gt_items]
        if gt_causes:
            c = Counter(gt_causes)
            return c.most_common(1)[0][0]
        return "semantic_gap"
    if fp_n >= 3 and matched <= 1:
        return "high_false_positive_rate"
    if fp_n > matched:
        return "precision_dominant_fp"
    if matched > 0 and matched < gt_n // 2:
        return "partial_recall_gap"
    return "mixed"


def _infer_gt_theme(error_text: str) -> str:
    text = str(error_text or "").lower()
    themes = [
        ("electromagnetism", ("magnetic", "flux", "induct", "faraday", "emf", "current loop", "rl circuit")),
        ("relativity_optics", ("light travel", "relativistic", "lorentz", "pinhole", "apparent length", "time delay")),
        ("orbital_gravity", ("orbital", "binary", "gravitational", "kepler", "collision rate", "self-gravit")),
        ("thermo_fluid", ("heat", "temperature", "pressure", "density", "phase", "latent", "fluid", "area")),
        ("mechanics", ("friction", "force", "torque", "momentum", "inertia", "velocity", "acceleration")),
        ("waves_optics", ("wavelength", "refraction", "interference", "evanescent", "optical path")),
        ("formula_algebra", ("dimension", "incorrect expression", "wrong formula", "factor of", "sqrt(")),
    ]
    for name, keys in themes:
        if any(k in text for k in keys):
            return name
    return "other"


def _build_by_rule_report(
    enriched: List[Dict[str, Any]],
    ds_by_id: Dict[str, Dict[str, Any]],
    ver_by_id: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    rule_fp: Counter = Counter()
    rule_fp_causes: Dict[str, Counter] = defaultdict(Counter)
    rule_near_miss: Counter = Counter()
    theme_fn: Counter = Counter()
    theme_fn_causes: Dict[str, Counter] = defaultdict(Counter)
    location_failure_samples: List[Dict[str, Any]] = []

    for row in enriched:
        sid = str(row.get("id") or "")
        gt_items = [g for g in (ds_by_id.get(sid, {}).get("physics_error_gt") or []) if isinstance(g, dict)]
        findings = [f for f in (ver_by_id.get(sid, {}).get("diagnostics") or []) if isinstance(f, dict)]
        matched_ids = {m.get("gt_error_id") for m in row.get("location_matches") or [] if isinstance(m, dict)}

        for g in gt_items:
            eid = str(g.get("error_id") or "")
            if eid in matched_ids:
                continue
            theme = _infer_gt_theme(str(g.get("error_text") or ""))
            cause, meta = _classify_unmatched_gt(g, findings)
            theme_fn[theme] += 1
            theme_fn_causes[theme][cause] += 1
            rule_near_miss[str(meta.get("best_pred_rule") or "none")] += 1
            if cause in {"location_failure", "semantic_near_miss"}:
                location_failure_samples.append(
                    {
                        "sample_id": sid,
                        "error_id": eid,
                        "cause": cause,
                        "theme": theme,
                        "best_pred_rule": meta.get("best_pred_rule"),
                        "best_token_overlap": meta.get("best_token_overlap"),
                    }
                )

        previews = row.get("unmatched_pred_locatable_preview") or []
        if previews:
            fp_iter = previews
        else:
            fp_iter = findings
        for item in fp_iter:
            if isinstance(item, dict) and "rule" in item and "message" not in item:
                pseudo = {
                    "rule": item.get("rule"),
                    "message": "",
                    "evidence": {"quote": item.get("quote"), "location": {"paragraph_index": item.get("paragraph_index")}},
                }
            else:
                pseudo = item if isinstance(item, dict) else {}
            cause, meta = _classify_fp_pred(pseudo, gt_items)
            rid = str(meta.get("rule") or pseudo.get("rule") or "unknown")
            rule_fp[rid] += 1
            rule_fp_causes[rid][cause] += 1

    top_fp_rules = [
        {
            "rule_id": rid,
            "fp_count": cnt,
            "causes": dict(rule_fp_causes[rid]),
        }
        for rid, cnt in rule_fp.most_common(25)
    ]
    top_missed_themes = [
        {"theme": theme, "fn_count": cnt, "causes": dict(theme_fn_causes[theme])}
        for theme, cnt in theme_fn.most_common()
    ]
    location_failure_samples.sort(
        key=lambda x: (-int(x.get("best_token_overlap") or 0), str(x.get("sample_id") or ""))
    )
    return {
        "top_fp_rules": top_fp_rules,
        "top_missed_gt_themes": top_missed_themes,
        "top_near_miss_rules": [{"rule_id": rid, "near_miss_count": cnt} for rid, cnt in rule_near_miss.most_common(25)],
        "location_failure_samples": location_failure_samples[:40],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze scale eval failure cases.")
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--verifier-results", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--by-rule-output",
        type=str,
        default="",
        help="Optional path for rule/theme aggregated failure report JSON.",
    )
    args = parser.parse_args()

    metrics = _load_json(Path(args.metrics))
    details = metrics.get("details") if isinstance(metrics, dict) else []
    if not isinstance(details, list):
        raise SystemExit("metrics must contain details array")

    verifier_rows = _load_json(Path(args.verifier_results))
    if not isinstance(verifier_rows, list):
        raise SystemExit("verifier results must be a JSON array")
    ver_by_id = {str(r.get("id") or ""): r for r in verifier_rows if isinstance(r, dict)}

    dataset_rows = _load_json(Path(args.dataset))
    if not isinstance(dataset_rows, list):
        raise SystemExit("dataset must be a JSON array")
    ds_by_id = {str(r.get("id") or ""): r for r in dataset_rows if isinstance(r, dict)}

    enriched: List[Dict[str, Any]] = []
    for d in details:
        if not isinstance(d, dict):
            continue
        sid = str(d.get("id") or "")
        gt_items = ds_by_id.get(sid, {}).get("physics_error_gt") or []
        gt_items = [g for g in gt_items if isinstance(g, dict)]
        findings = ver_by_id.get(sid, {}).get("diagnostics") or []
        findings = [f for f in findings if isinstance(f, dict)]
        primary = _sample_primary_cause(d, gt_items, findings)
        enriched.append(
            {
                **d,
                "primary_failure_cause": primary,
                "gt_count": len(gt_items),
                "pred_count": len(findings),
                "fp_count": int(d.get("unmatched_pred_locatable_count") or 0),
                "fn_count": int(d.get("gt_error_count") or 0) - int(d.get("matched_error_count") or 0),
            }
        )

    zero_match = [
        x for x in enriched if int(x.get("matched_error_count") or 0) == 0 and int(x.get("gt_error_count") or 0) > 0
    ]
    zero_match.sort(key=lambda x: (-int(x.get("gt_error_count") or 0), -int(x.get("pred_count") or 0)))
    high_fp = [x for x in enriched if int(x.get("fp_count") or 0) > 0]
    high_fp.sort(key=lambda x: (-int(x.get("fp_count") or 0), -int(x.get("pred_count") or 0)))

    top_k = max(1, int(args.top_k))
    zero_pick = zero_match[:top_k]
    fp_pick = high_fp[:top_k]

    zero_cases: List[Dict[str, Any]] = []
    gt_cause_counter: Counter = Counter()
    sample_cause_counter: Counter = Counter()

    for row in zero_pick:
        sid = str(row.get("id") or "")
        gt_items = ds_by_id.get(sid, {}).get("physics_error_gt") or []
        gt_items = [g for g in gt_items if isinstance(g, dict)]
        findings = ver_by_id.get(sid, {}).get("diagnostics") or []
        findings = [f for f in findings if isinstance(f, dict)]
        gt_breakdown = []
        for g in gt_items:
            cause, meta = _classify_unmatched_gt(g, findings)
            gt_cause_counter[cause] += 1
            gt_breakdown.append(
                {
                    "error_id": g.get("error_id"),
                    "cause": cause,
                    "error_preview": str(g.get("error_text") or "")[:160],
                    **meta,
                }
            )
        sample_cause_counter[row["primary_failure_cause"]] += 1
        zero_cases.append(
            {
                "id": sid,
                "gt_count": row.get("gt_count"),
                "pred_count": row.get("pred_count"),
                "matched_count": row.get("matched_error_count"),
                "primary_failure_cause": row.get("primary_failure_cause"),
                "gt_failure_breakdown": gt_breakdown,
                "pred_preview": [
                    {"rule": f.get("rule"), "message": str(f.get("message") or "")[:160]}
                    for f in findings[:4]
                ],
            }
        )

    fp_cases: List[Dict[str, Any]] = []
    fp_cause_counter: Counter = Counter()
    fp_sample_cause_counter: Counter = Counter()

    for row in fp_pick:
        sid = str(row.get("id") or "")
        gt_items = ds_by_id.get(sid, {}).get("physics_error_gt") or []
        gt_items = [g for g in gt_items if isinstance(g, dict)]
        findings = ver_by_id.get(sid, {}).get("diagnostics") or []
        findings = [f for f in findings if isinstance(f, dict)]

        matched_ids = {m.get("gt_error_id") for m in row.get("location_matches") or [] if isinstance(m, dict)}
        unmatched_preds = []
        # Use preview from metrics when available; otherwise all findings are potential FP if not in matched set
        previews = row.get("unmatched_pred_locatable_preview") or []
        if previews:
            for p in previews:
                if not isinstance(p, dict):
                    continue
                pseudo = {
                    "rule": p.get("rule"),
                    "message": "",
                    "evidence": {"quote": p.get("quote"), "location": {"paragraph_index": p.get("paragraph_index")}},
                }
                cause, meta = _classify_fp_pred(pseudo, gt_items)
                fp_cause_counter[cause] += 1
                unmatched_preds.append({"cause": cause, **meta, "quote": p.get("quote", "")[:120]})
        else:
            for f in findings:
                cause, meta = _classify_fp_pred(f, gt_items)
                fp_cause_counter[cause] += 1
                unmatched_preds.append({"cause": cause, **meta})

        fp_sample_cause_counter[row["primary_failure_cause"]] += 1
        fp_cases.append(
            {
                "id": sid,
                "gt_count": row.get("gt_count"),
                "pred_count": row.get("pred_count"),
                "matched_count": row.get("matched_error_count"),
                "fp_count": row.get("fp_count"),
                "primary_failure_cause": row.get("primary_failure_cause"),
                "fp_breakdown": unmatched_preds[:6],
            }
        )

    by_rule = _build_by_rule_report(enriched, ds_by_id, ver_by_id)

    out = {
        "source": {
            "metrics": args.metrics,
            "verifier_results": args.verifier_results,
            "dataset": args.dataset,
        },
        "summary": {
            "total_samples": len(enriched),
            "zero_match_high_gt_selected": len(zero_cases),
            "high_fp_selected": len(fp_cases),
            "zero_match_sample_causes": dict(sample_cause_counter),
            "zero_match_gt_causes": dict(gt_cause_counter),
            "fp_item_causes": dict(fp_cause_counter),
            "high_fp_sample_causes": dict(fp_sample_cause_counter),
        },
        "taxonomy": {
            "no_detection": "规则未触发或未发布 diagnostic",
            "semantic_gap": "有输出但与 GT 错误点/token 几乎无关",
            "semantic_near_miss": "语义部分重叠但未达 location 匹配",
            "location_failure": "同段落/高 token 重叠但 span/region 未对齐",
            "rule_too_broad": "规则触发宽泛，报相邻或泛化问题",
            "gt_granularity_mismatch": "GT 很细但规则输出很少",
            "irrelevant_trigger": "触发点与所有 GT 无关",
            "high_false_positive_rate": "样本级：FP 多、命中少",
            "precision_dominant_fp": "样本级：FP 数高于 TP",
            "partial_recall_gap": "样本级：部分命中但大量 GT 漏检",
        },
        "zero_match_high_gt_cases": zero_cases,
        "high_fp_cases": fp_cases,
        "by_rule": by_rule,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    by_rule_path = str(args.by_rule_output or "").strip()
    if not by_rule_path:
        by_rule_path = str(out_path.with_name("failure_analysis_by_rule.json"))
    Path(by_rule_path).write_text(json.dumps(by_rule, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**out["summary"], "by_rule_output": by_rule_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
