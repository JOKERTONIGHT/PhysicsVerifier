from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm(s: Optional[str]) -> str:
    return (s or "").strip() or "<unknown>"


_INCONCLUSIVE_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("missing_required_symbols", re.compile(r"no parsed equation contains required symbols", re.I)),
    ("no_equation_relates_variables", re.compile(r"no equation relates the provided variables", re.I)),
    ("canonical_unparseable", re.compile(r"canonical equations could not be parsed", re.I)),
    ("no_equivalent_found", re.compile(r"no extracted equation is equivalent to the canonical form", re.I)),
    ("other", re.compile(r".*", re.S)),
]


def _classify_msg(msg: Optional[str]) -> str:
    if not msg:
        return "empty"
    for name, pat in _INCONCLUSIVE_PATTERNS:
        if pat.search(msg):
            return name
    return "other"


@dataclass
class TopicAgg:
    topic: str
    samples: int = 0
    samples_with_symbolic: int = 0
    symbolic_total: int = 0
    symbolic_pass: int = 0
    symbolic_fail: int = 0
    symbolic_inconclusive: int = 0
    checked_diagnostics: int = 0
    suppressed_diagnostics: int = 0


@dataclass
class SpecAgg:
    spec_id: str
    primitive: str
    title: str
    total: int = 0
    pass_count: int = 0
    fail_count: int = 0
    inconclusive_count: int = 0
    topics_top: List[Tuple[str, int]] = None
    reasons_top: List[Tuple[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["topics_top"] = self.topics_top or []
        d["reasons_top"] = self.reasons_top or []
        return d


def _load_catalog_index(path: Path) -> Dict[str, List[Tuple[str, str, str]]]:
    """Return spec_id -> list of (domain, topic, primitive)."""
    if not path.exists():
        return {}
    try:
        data = _load_json(path)
    except Exception:
        return {}

    idx: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for dom in (data.get("domains") or []):
        dname = _norm(dom.get("name"))
        for top in (dom.get("topics") or []):
            tname = _norm(top.get("name"))
            for chk in (top.get("checks") or []):
                sid = chk.get("spec_id")
                if not sid:
                    continue
                idx[str(sid)].append((dname, tname, _norm(chk.get("primitive"))))
    return dict(idx)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze symbolic_audit_*.json")
    ap.add_argument("--audit", required=True, help="Path to results/symbolic_audit_*.json")
    ap.add_argument("--outdir", default="results/symbolic_audit_analysis")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--catalog", default="catalogs/symbolic_catalog.json")
    args = ap.parse_args()

    audit_path = Path(args.audit)
    outdir = Path(args.outdir)
    topk = int(args.topk)

    data = _load_json(audit_path)
    if not isinstance(data, list):
        raise SystemExit(f"Audit file must be a JSON array. got {type(data)}")

    catalog_index = _load_catalog_index(Path(args.catalog))

    total_samples = len(data)

    # Global counters
    symbolic_result_counter = Counter()
    primitive_counter = Counter()
    reconciliation_counter = Counter()  # supported/inconclusive from checked_diagnostics
    checked_severity_counter = Counter()

    checked_rule_counter = Counter()
    suppressed_rule_counter = Counter()
    suppressed_spec_counter = Counter()

    # Topic agg
    topic_agg: Dict[str, TopicAgg] = {}

    # Spec agg
    spec_total = Counter()
    spec_result = defaultdict(Counter)  # spec_id -> pass/fail/inconclusive
    spec_primitive = {}
    spec_title = {}
    spec_topic_dist = defaultdict(Counter)
    spec_reason_dist = defaultdict(Counter)
    spec_topics_unique = defaultdict(set)

    # Quality checks
    duplicate_symbolic_entries = 0
    total_symbolic_entries = 0
    summary_mismatch_examples: List[Dict[str, Any]] = []

    # Coverage linkage
    referenced_spec_ids = Counter()  # from checked_diagnostics + suppressed_diagnostics
    orphan_symbolic_entries: List[Dict[str, Any]] = []

    for sample in data:
        sid = _norm(sample.get("id"))
        topic = _norm(sample.get("topic"))

        ta = topic_agg.get(topic)
        if ta is None:
            ta = TopicAgg(topic=topic)
            topic_agg[topic] = ta
        ta.samples += 1

        checked = sample.get("checked_diagnostics") or []
        suppressed = sample.get("suppressed_diagnostics") or []
        sym_checks = sample.get("symbolic_checks") or []
        sym_summary = sample.get("symbolic_summary") or {}

        ta.checked_diagnostics += len(checked)
        ta.suppressed_diagnostics += len(suppressed)

        # reconcile status counts from checked_diagnostics
        for d in checked:
            if not isinstance(d, dict):
                continue
            checked_rule_counter[_norm(d.get("rule"))] += 1
            checked_severity_counter[_norm(d.get("severity"))] += 1

            rec = d.get("symbolic_reconciliation")
            if isinstance(rec, dict) and rec.get("status"):
                reconciliation_counter[_norm(rec.get("status"))] += 1

            for x in (d.get("symbolic_cross_checks") or []):
                referenced_spec_ids[_norm(str(x))] += 1

        for s in suppressed:
            if not isinstance(s, dict):
                continue
            for x in (s.get("spec_ids") or []):
                suppressed_spec_counter[_norm(str(x))] += 1
                referenced_spec_ids[_norm(str(x))] += 1
            od = s.get("original_diagnostic")
            if isinstance(od, dict):
                suppressed_rule_counter[_norm(od.get("rule"))] += 1

        # symbolic checks
        if sym_checks:
            ta.samples_with_symbolic += 1

        # prefer summary if available, but also compute and verify
        computed = Counter()
        seen_fp = Counter()
        sample_referenced_set = set(referenced_spec_ids.keys())  # global set for now (cheaper); refine below

        # Build a sample-level referenced set precisely
        sample_referenced_set = set()
        for d in checked:
            if isinstance(d, dict):
                for x in (d.get("symbolic_cross_checks") or []):
                    sample_referenced_set.add(_norm(str(x)))
        for s in suppressed:
            if isinstance(s, dict):
                for x in (s.get("spec_ids") or []):
                    sample_referenced_set.add(_norm(str(x)))

        for e in sym_checks:
            if not isinstance(e, dict):
                continue
            total_symbolic_entries += 1

            spid = _norm(e.get("spec_id"))
            prim = _norm(e.get("primitive"))
            title = _norm(e.get("title"))
            res = _norm(e.get("result")).lower()

            primitive_counter[prim] += 1
            symbolic_result_counter[res] += 1

            spec_total[spid] += 1
            spec_result[spid][res] += 1
            spec_primitive.setdefault(spid, prim)
            spec_title.setdefault(spid, title)
            spec_topic_dist[spid][topic] += 1
            spec_topics_unique[spid].add(topic)

            computed[res] += 1

            if res == "inconclusive":
                spec_reason_dist[spid][_classify_msg(e.get("message"))] += 1

            fp = (spid, prim, res, (e.get("evidence") or "")[:120], (e.get("message") or "")[:120])
            seen_fp[fp] += 1

            if spid not in sample_referenced_set:
                orphan_symbolic_entries.append(
                    {
                        "id": sid,
                        "topic": topic,
                        "spec_id": spid,
                        "primitive": prim,
                        "result": res,
                        "message": e.get("message"),
                    }
                )

        for fp, c in seen_fp.items():
            if c > 1:
                duplicate_symbolic_entries += (c - 1)

        # summary mismatch check
        if isinstance(sym_summary, dict) and sym_checks:
            exp_total = int(sym_summary.get("total", 0) or 0)
            exp_pass = int(sym_summary.get("pass", 0) or 0)
            exp_fail = int(sym_summary.get("fail", 0) or 0)
            exp_inc = int(sym_summary.get("inconclusive", 0) or 0)

            got_total = sum(computed.values())
            got_pass = computed.get("pass", 0)
            got_fail = computed.get("fail", 0)
            got_inc = computed.get("inconclusive", 0)

            if (exp_total, exp_pass, exp_fail, exp_inc) != (got_total, got_pass, got_fail, got_inc):
                if len(summary_mismatch_examples) < 50:
                    summary_mismatch_examples.append(
                        {
                            "id": sid,
                            "topic": topic,
                            "symbolic_summary": sym_summary,
                            "computed": {
                                "total": got_total,
                                "pass": got_pass,
                                "fail": got_fail,
                                "inconclusive": got_inc,
                            },
                        }
                    )

        # aggregate topic counters using the summary (since it's already what the system emitted)
        if isinstance(sym_summary, dict):
            ta.symbolic_total += int(sym_summary.get("total", 0) or 0)
            ta.symbolic_pass += int(sym_summary.get("pass", 0) or 0)
            ta.symbolic_fail += int(sym_summary.get("fail", 0) or 0)
            ta.symbolic_inconclusive += int(sym_summary.get("inconclusive", 0) or 0)

    # Build spec aggs
    spec_aggs: List[SpecAgg] = []
    for spid, total in spec_total.items():
        agg = SpecAgg(
            spec_id=spid,
            primitive=spec_primitive.get(spid, "<unknown>"),
            title=spec_title.get(spid, "<unknown>"),
            total=total,
            pass_count=spec_result[spid].get("pass", 0),
            fail_count=spec_result[spid].get("fail", 0),
            inconclusive_count=spec_result[spid].get("inconclusive", 0),
        )
        agg.topics_top = spec_topic_dist[spid].most_common(10)
        agg.reasons_top = spec_reason_dist[spid].most_common(10)
        spec_aggs.append(agg)

    spec_aggs.sort(key=lambda x: x.total, reverse=True)

    # Topic aggs list
    topic_aggs = sorted(topic_agg.values(), key=lambda x: x.symbolic_total, reverse=True)

    # Cross-topic specs (heuristic)
    cross_topic_specs = sorted(
        ((spid, len(spec_topics_unique[spid]), spec_total[spid]) for spid in spec_total.keys()),
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )

    used_spec_ids = set(spec_total.keys())
    catalog_spec_ids = set(catalog_index.keys())
    unknown_specs = sorted(list(used_spec_ids - catalog_spec_ids))
    unused_catalog_specs = sorted(list(catalog_spec_ids - used_spec_ids))

    out_summary = {
        "input": str(audit_path),
        "catalog": str(Path(args.catalog)),
        "global": {
            "total_samples": total_samples,
            "symbolic_entries_total": total_symbolic_entries,
            "symbolic_entries_duplicate": duplicate_symbolic_entries,
            "duplicate_ratio": (duplicate_symbolic_entries / total_symbolic_entries) if total_symbolic_entries else 0.0,
            "symbolic_result_counts": dict(symbolic_result_counter),
            "primitive_counts": dict(primitive_counter),
            "reconciliation_status_counts_checked_only": dict(reconciliation_counter),
            "checked_by_severity": dict(checked_severity_counter),
            "checked_diagnostics_total": sum(t.checked_diagnostics for t in topic_aggs),
            "suppressed_diagnostics_total": sum(t.suppressed_diagnostics for t in topic_aggs),
            "orphan_symbolic_entries_count": len(orphan_symbolic_entries),
            "summary_mismatch_count": len(summary_mismatch_examples),
            "unknown_specs_used_not_in_catalog": len(unknown_specs),
            "unused_specs_in_catalog": len(unused_catalog_specs),
        },
        "top_specs_by_usage": [s.to_dict() for s in spec_aggs[:topk]],
        "top_checked_rules": checked_rule_counter.most_common(topk),
        "top_suppressed_rules": suppressed_rule_counter.most_common(topk),
        "top_suppressed_specs": suppressed_spec_counter.most_common(topk),
        "topic_stats": [asdict(t) for t in topic_aggs],
        "cross_topic_specs": [
            {"spec_id": spid, "unique_topics": ut, "total_occurrences": occ, "catalog_home": catalog_index.get(spid, [])}
            for spid, ut, occ in cross_topic_specs[:topk]
        ],
        "unknown_specs": unknown_specs[: min(len(unknown_specs), 200)],
        "unused_catalog_specs": unused_catalog_specs[: min(len(unused_catalog_specs), 200)],
        "orphans_examples": orphan_symbolic_entries[: min(len(orphan_symbolic_entries), 50)],
        "summary_mismatch_examples": summary_mismatch_examples,
    }

    _dump_json(outdir / "summary.json", out_summary)
    _dump_json(outdir / "spec_stats_full.json", [s.to_dict() for s in spec_aggs])

    # Write a compact markdown report
    g = out_summary["global"]
    lines: List[str] = []
    lines.append("# Symbolic Audit 统计报告")
    lines.append("")
    lines.append(f"- 输入: `{audit_path}`")
    lines.append(f"- 样本数: {g['total_samples']}")
    lines.append(f"- 符号检查条目总数: {g['symbolic_entries_total']}")
    lines.append(
        f"- 同一样本内重复符号条目数: {g['symbolic_entries_duplicate']} (比例 {g['duplicate_ratio']:.2%})"
    )
    lines.append("")

    lines.append("## 符号检查结果分布")
    lines.append(f"- pass/fail/inconclusive: {g['symbolic_result_counts']}")
    lines.append("")

    lines.append("## Primitive 使用分布")
    lines.append(f"- {g['primitive_counts']}")
    lines.append("")

    lines.append("## 复查对齐（仅 checked_diagnostics）")
    lines.append(f"- reconciliation status: {g['reconciliation_status_counts_checked_only']}")
    lines.append("")

    lines.append("## 抑制（suppress）")
    lines.append(f"- 被 suppress 的诊断总数: {g['suppressed_diagnostics_total']}")
    if out_summary["top_suppressed_rules"]:
        lines.append("- suppress 最多的规则(Top 10):")
        for rid, c in out_summary["top_suppressed_rules"][:10]:
            lines.append(f"  - {rid}: {c}")
    if out_summary["top_suppressed_specs"]:
        lines.append("- suppress 最多的 spec(Top 10):")
        for spid, c in out_summary["top_suppressed_specs"][:10]:
            lines.append(f"  - {spid}: {c}")
    lines.append("")

    lines.append("## Spec 使用量 Top")
    for s in out_summary["top_specs_by_usage"][:10]:
        lines.append(
            f"- {s['spec_id']} ({s['primitive']}): total={s['total']} pass={s['pass_count']} fail={s['fail_count']} inconclusive={s['inconclusive_count']}"
        )
        if s.get("topics_top"):
            lines.append(f"  - topics_top: {s['topics_top'][:5]}")
        if s.get("reasons_top"):
            lines.append(f"  - inconclusive_reasons_top: {s['reasons_top'][:5]}")
    lines.append("")

    lines.append("## 跨 Topic 频繁出现的 spec (Top)")
    for x in out_summary["cross_topic_specs"][:10]:
        home = x.get("catalog_home") or []
        home_str = "; ".join([f"{d}/{t}" for d, t, _ in home]) if home else "<not-in-catalog>"
        lines.append(
            f"- {x['spec_id']}: unique_topics={x['unique_topics']} total_occurrences={x['total_occurrences']} home={home_str}"
        )
    lines.append("")

    lines.append("## 数据质量检查")
    lines.append(f"- orphan symbolic entries（不被任何诊断引用）: {g['orphan_symbolic_entries_count']}")
    lines.append(f"- symbolic_summary 与 computed 不一致的样本数: {g['summary_mismatch_count']}")
    lines.append(f"- 使用但不在 catalog 的 spec 数: {g['unknown_specs_used_not_in_catalog']}")
    lines.append(f"- catalog 中存在但未被使用的 spec 数: {g['unused_specs_in_catalog']}")

    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] wrote: {outdir / 'summary.json'}")
    print(f"[ok] wrote: {outdir / 'spec_stats_full.json'}")
    print(f"[ok] wrote: {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
