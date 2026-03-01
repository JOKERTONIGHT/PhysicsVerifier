from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _norm(s: Optional[str]) -> str:
    return (s or "").strip() or "<unknown>"


@dataclass
class TopicSummary:
    domain: str
    topic: str
    total_specs: int
    primitives: Dict[str, int]
    missing_required_symbols: int
    missing_match_rule_ids: int
    missing_match_keywords: int


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze catalogs/symbolic_catalog.json")
    ap.add_argument("--catalog", default="catalogs/symbolic_catalog.json")
    ap.add_argument("--outdir", default="results/symbolic_catalog_analysis")
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    catalog_path = Path(args.catalog)
    outdir = Path(args.outdir)
    topk = int(args.topk)

    data = _load_json(catalog_path)
    if not isinstance(data, dict):
        raise SystemExit(f"Catalog must be a JSON object. got {type(data)}")

    # Global counters
    total_specs = 0
    domain_counter = Counter()
    topic_counter = Counter()
    primitive_counter = Counter()

    required_symbols_missing = 0
    match_rule_ids_missing = 0
    match_keywords_missing = 0

    spec_id_counter = Counter()

    topic_summaries: List[TopicSummary] = []

    for dom in (data.get("domains") or []):
        domain_name = _norm(dom.get("name"))
        for top in (dom.get("topics") or []):
            topic_name = _norm(top.get("name"))
            checks = top.get("checks") or []

            local_total = 0
            local_primitive = Counter()
            local_missing_required = 0
            local_missing_rule_ids = 0
            local_missing_keywords = 0

            for chk in checks:
                if not isinstance(chk, dict):
                    continue
                spec_id = _norm(chk.get("spec_id"))
                spec_id_counter[spec_id] += 1

                prim = _norm(chk.get("primitive"))
                params = chk.get("params") if isinstance(chk.get("params"), dict) else {}

                req = params.get("required_symbols")
                if not req:
                    required_symbols_missing += 1
                    local_missing_required += 1

                mr = chk.get("match_rule_ids")
                if not mr:
                    match_rule_ids_missing += 1
                    local_missing_rule_ids += 1

                mk = chk.get("match_keywords")
                if not mk:
                    match_keywords_missing += 1
                    local_missing_keywords += 1

                total_specs += 1
                local_total += 1
                domain_counter[domain_name] += 1
                topic_counter[f"{domain_name} / {topic_name}"] += 1
                primitive_counter[prim] += 1
                local_primitive[prim] += 1

            topic_summaries.append(
                TopicSummary(
                    domain=domain_name,
                    topic=topic_name,
                    total_specs=local_total,
                    primitives=dict(local_primitive),
                    missing_required_symbols=local_missing_required,
                    missing_match_rule_ids=local_missing_rule_ids,
                    missing_match_keywords=local_missing_keywords,
                )
            )

    duplicate_spec_ids = {sid: c for sid, c in spec_id_counter.items() if c > 1}

    # Sort topic summaries by spec count desc
    topic_summaries.sort(key=lambda x: x.total_specs, reverse=True)

    summary = {
        "catalog": str(catalog_path),
        "global": {
            "total_specs": total_specs,
            "domains": dict(domain_counter),
            "primitives": dict(primitive_counter),
            "topics_total": len(topic_summaries),
            "spec_ids_unique": len(spec_id_counter),
            "spec_ids_duplicate_count": len(duplicate_spec_ids),
            "missing_required_symbols": required_symbols_missing,
            "missing_match_rule_ids": match_rule_ids_missing,
            "missing_match_keywords": match_keywords_missing,
        },
        "topics": [asdict(t) for t in topic_summaries],
        "top_topics_by_specs": [asdict(t) for t in topic_summaries[:topk]],
        "duplicate_spec_ids": dict(sorted(duplicate_spec_ids.items(), key=lambda kv: kv[1], reverse=True)),
    }

    _dump_json(outdir / "summary.json", summary)

    # Human-readable report
    lines: List[str] = []
    g = summary["global"]
    lines.append("# Symbolic Catalog 统计报告")
    lines.append("")
    lines.append(f"- catalog: `{catalog_path}`")
    lines.append(f"- spec 总数: {g['total_specs']}")
    lines.append(f"- domain 数: {len(g['domains'])}")
    lines.append(f"- topic 数: {g['topics_total']}")
    lines.append(f"- spec_id 唯一数: {g['spec_ids_unique']}")
    lines.append(f"- spec_id 重复数: {g['spec_ids_duplicate_count']}")
    lines.append("")

    lines.append("## Domain 分布")
    for k, v in domain_counter.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## Primitive 分布")
    for k, v in primitive_counter.most_common():
        lines.append(f"- {k}: {v}")
    lines.append("")

    lines.append("## 质量检查（可能导致误匹配/误复查）")
    lines.append(f"- 缺少 params.required_symbols 的 spec 数: {g['missing_required_symbols']}")
    lines.append(f"- 缺少 match_rule_ids 的 spec 数: {g['missing_match_rule_ids']}")
    lines.append(f"- 缺少 match_keywords 的 spec 数: {g['missing_match_keywords']}")
    lines.append("")

    lines.append("## spec 数最多的 topic (Top)")
    for t in topic_summaries[: min(topk, 20)]:
        lines.append(f"- {t.domain} / {t.topic}: {t.total_specs} (primitives={t.primitives})")
    lines.append("")

    if duplicate_spec_ids:
        lines.append("## 重复 spec_id（跨 topic/domain 重名）")
        for sid, c in sorted(duplicate_spec_ids.items(), key=lambda kv: kv[1], reverse=True)[: min(topk, 30)]:
            lines.append(f"- {sid}: {c}")
        lines.append("")

    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[ok] wrote: {outdir / 'summary.json'}")
    print(f"[ok] wrote: {outdir / 'report.md'}")


if __name__ == "__main__":
    main()
