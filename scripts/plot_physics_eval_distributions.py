#!/usr/bin/env python3
"""Generate distribution charts for rule library, error-level eval set, and matched errors."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Chinese display names for catalog domains / error types
# Short English labels (server may lack CJK fonts)
DOMAIN_LABEL = {
    "Mechanics": "Mechanics",
    "Electromagnetism": "EM",
    "Thermodynamics & Statistical Physics": "Thermo/Stat",
    "Modern Physics": "Modern",
    "Optics": "Optics",
    "Experimental Physics": "Experiment",
}

ERROR_TYPE_LABEL = {
    "concept": "Concept",
    "logic": "Logic",
    "calculation": "Calculation",
    "modeling": "Modeling",
    "units": "Units",
}

ERROR_TYPE_ORDER = ["concept", "logic", "calculation", "modeling", "units"]

# Keyword heuristics aligned with thesis taxonomy (concept / logic / calculation / modeling / units)
_ERROR_TYPE_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    (
        "units",
        re.compile(
            r"\b(unit|units|dimensional|dimension|si\b|kg|meter|metre|cm\b|mm\b|"
            r"quantity|量纲|单位)\b",
            re.I,
        ),
    ),
    (
        "calculation",
        re.compile(
            r"\b(numerical|arithmetic|compute|calculat|round|approximat|"
            r"≈|=\s*[\d.]|\d+\.\d+|value\s+is\s+wrong|mistake\s+in\s+the\s+calculation)\b",
            re.I,
        ),
    ),
    (
        "modeling",
        re.compile(
            r"\b(model|assumption|approximat|boundary|frame\s+of\s+reference|reference\s+frame|"
            r"idealiz|scenario|situation|applicab|valid\s+when|cannot\s+apply)\b",
            re.I,
        ),
    ),
    (
        "concept",
        re.compile(
            r"\b(misunderstand|concept|definition|physical\s+meaning|interpret|"
            r"confus|incorrectly\s+assumes|wrong\s+physical)\b",
            re.I,
        ),
    ),
    (
        "logic",
        re.compile(
            r"\b(therefore|thus|implies|inconsistent|contradict|missing\s+step|"
            r"does\s+not\s+follow|chain|reasoning|logic)\b",
            re.I,
        ),
    ),
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def classify_error_type(error_text: str) -> str:
    text = str(error_text or "")
    for label, pat in _ERROR_TYPE_PATTERNS:
        if pat.search(text):
            return label
    return "logic"


def build_topic_to_domain(catalog: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for domain in catalog.get("domains") or []:
        dname = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics") or []:
            tname = str(topic.get("name") or "")
            if tname:
                out[tname] = dname
    return out


def build_rule_index(catalog: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """Map rule_id -> domain, topic, error_type."""
    idx: Dict[str, Dict[str, str]] = {}
    for domain in catalog.get("domains") or []:
        dname = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics") or []:
            tname = str(topic.get("name") or "Unknown")
            for rule in topic.get("rules") or []:
                if not isinstance(rule, dict):
                    continue
                rid = str(rule.get("rule_id") or rule.get("id") or "").strip()
                if not rid:
                    continue
                idx[rid] = {
                    "domain": dname,
                    "topic": tname,
                    "error_type": str(rule.get("error_type") or "logic"),
                }
    return idx


def count_rules_by_domain_topic(catalog: Dict[str, Any]) -> Tuple[Counter, Counter]:
    by_domain: Counter = Counter()
    by_topic: Counter = Counter()
    for domain in catalog.get("domains") or []:
        dname = str(domain.get("name") or "Unknown")
        for topic in domain.get("topics") or []:
            tname = str(topic.get("name") or "Unknown")
            n = len([r for r in (topic.get("rules") or []) if isinstance(r, dict)])
            by_domain[dname] += n
            by_topic[(dname, tname)] = n
    return by_domain, by_topic


def iter_gt_errors_with_domain(
    dataset: List[Dict[str, Any]],
    sample_domain: Dict[str, str],
) -> List[Tuple[str, str, str]]:
    """Yield (sample_id, domain, error_type) for each GT error."""
    rows: List[Tuple[str, str, str]] = []
    for item in dataset:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "")
        domain = sample_domain.get(sid, "Unknown")
        for err in item.get("physics_error_gt") or []:
            if not isinstance(err, dict):
                continue
            if not bool(err.get("locatable_valid")):
                continue
            etype = classify_error_type(str(err.get("error_text") or ""))
            rows.append((sid, domain, etype))
    return rows


def parse_rule_id_from_pred_text(pred_text: str) -> str:
    head = str(pred_text or "").split("|", 1)[0].strip()
    return head


def collect_matched_errors(
    metrics: Dict[str, Any],
    rule_index: Dict[str, Dict[str, str]],
    sample_domain: Dict[str, str],
    use_rule_type: bool = True,
) -> List[Tuple[str, str, str]]:
    """Yield (gt_error_id, domain, error_type) for each location-matched GT error."""
    rows: List[Tuple[str, str, str]] = []
    for detail in metrics.get("details") or []:
        if not isinstance(detail, dict):
            continue
        sid = str(detail.get("id") or "")
        domain = sample_domain.get(sid, "Unknown")
        for match in detail.get("location_matches") or []:
            if not isinstance(match, dict):
                continue
            gid = str(match.get("gt_error_id") or "")
            pred_text = str(match.get("pred_text") or "")
            rule_id = parse_rule_id_from_pred_text(pred_text)
            meta = rule_index.get(rule_id, {})
            if use_rule_type and meta.get("error_type"):
                etype = str(meta["error_type"])
                dom = str(meta.get("domain") or domain)
            else:
                etype = classify_error_type(str(match.get("gt_error_text") or ""))
                dom = domain
            rows.append((gid, dom, etype))
    return rows


def _setup_matplotlib() -> None:
    plt.rcParams["axes.unicode_minus"] = False


def plot_rule_library(by_domain: Counter, by_topic: Counter, out_dir: Path) -> None:
    _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    domains = [d for d, _ in by_domain.most_common()]
    counts = [by_domain[d] for d in domains]
    labels = [DOMAIN_LABEL.get(d, d) for d in domains]
    colors = plt.cm.Set2(np.linspace(0, 1, len(domains)))
    axes[0].bar(range(len(domains)), counts, color=colors, edgecolor="white", linewidth=0.6)
    axes[0].set_xticks(range(len(domains)))
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Leaf rules")
    axes[0].set_title(f"Rule library: rules per domain (n={sum(counts)})")
    for i, v in enumerate(counts):
        axes[0].text(i, v + 2, str(v), ha="center", va="bottom", fontsize=9)

    top_n = 18
    topic_items = sorted(by_topic.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))[:top_n]
    y_labels = [f"{DOMAIN_LABEL.get(d, d)[:6]} / {t[:32]}{'…' if len(t) > 32 else ''}" for (d, t), _ in topic_items]
    t_counts = [c for _, c in topic_items]
    y_pos = np.arange(len(topic_items))
    axes[1].barh(y_pos, t_counts, color=plt.cm.Pastel1(np.linspace(0, 1, len(topic_items))))
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(y_labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Leaf rules")
    axes[1].set_title(f"Rule library: top {top_n} topics by rule count")
    for i, v in enumerate(t_counts):
        axes[1].text(v + 0.3, i, str(v), va="center", fontsize=8)

    fig.tight_layout()
    path = out_dir / "01_rule_library_domain_topic.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def _domain_type_matrix(rows: List[Tuple[str, str, str]]) -> Tuple[List[str], List[str], np.ndarray]:
    domain_counts: Counter = Counter()
    for _, dom, _ in rows:
        domain_counts[dom] += 1
    domains = [d for d, _ in domain_counts.most_common()]
    types = ERROR_TYPE_ORDER
    mat = np.zeros((len(domains), len(types)), dtype=int)
    for _, dom, etype in rows:
        if dom not in domains:
            domains.append(dom)
        if etype not in types:
            etype = "logic"
        i = domains.index(dom)
        j = types.index(etype)
        mat[i, j] += 1
    return domains, types, mat


def plot_domain_type_heatmap(
    rows: List[Tuple[str, str, str]],
    title: str,
    out_path: Path,
    *,
    annotate_total: bool = True,
) -> None:
    _setup_matplotlib()
    domains, types, mat = _domain_type_matrix(rows)
    if mat.size == 0:
        print(f"Skip empty chart: {out_path}")
        return

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(domains) + 2)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([ERROR_TYPE_LABEL.get(t, t) for t in types], fontsize=10)
    ax.set_yticks(range(len(domains)))
    ax.set_yticklabels([DOMAIN_LABEL.get(d, d) for d in domains], fontsize=9)
    ax.set_title(f"{title}（n={len(rows)}）")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = int(mat[i, j])
            if v > 0:
                color = "white" if v > mat.max() * 0.55 else "black"
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9, color=color)

    if annotate_total:
        row_totals = mat.sum(axis=1)
        for i, tot in enumerate(row_totals):
            ax.text(len(types) + 0.15, i, f"Σ{int(tot)}", va="center", fontsize=9, color="#444")

    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_stacked_domain_bars(
    rows: List[Tuple[str, str, str]],
    title: str,
    out_path: Path,
) -> None:
    _setup_matplotlib()
    domains, types, mat = _domain_type_matrix(rows)
    if mat.size == 0:
        return

    x = np.arange(len(domains))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottom = np.zeros(len(domains))
    cmap = plt.cm.Set2(np.linspace(0, 1, len(types)))
    for j, etype in enumerate(types):
        vals = mat[:, j]
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=ERROR_TYPE_LABEL.get(etype, etype),
            color=cmap[j],
            edgecolor="white",
            linewidth=0.4,
        )
        bottom = bottom + vals

    ax.set_xticks(x)
    ax.set_xticklabels([DOMAIN_LABEL.get(d, d) for d in domains], fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title(f"{title}（n={len(rows)}）")
    ax.legend(loc="upper right", fontsize=9)
    for i, total in enumerate(bottom):
        ax.text(i, total + 1, str(int(total)), ha="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_stats_json(
    out_dir: Path,
    by_domain_rules: Counter,
    by_topic_rules: Counter,
    eval_rows: List[Tuple[str, str, str]],
    matched_rows: List[Tuple[str, str, str]],
) -> None:
    def summarize(rows: List[Tuple[str, str, str]]) -> Dict[str, Any]:
        c_dom: Counter = Counter()
        c_type: Counter = Counter()
        c_cross: Counter = Counter()
        for _, dom, etype in rows:
            c_dom[dom] += 1
            c_type[etype] += 1
            c_cross[(dom, etype)] += 1
        return {
            "total": len(rows),
            "by_domain": dict(c_dom),
            "by_error_type": dict(c_type),
            "by_domain_and_type": {f"{d}|{t}": n for (d, t), n in c_cross.items()},
        }

    payload = {
        "rule_library": {
            "total_rules": sum(by_domain_rules.values()),
            "by_domain": dict(by_domain_rules),
            "by_topic": {f"{d}::{t}": n for (d, t), n in by_topic_rules.items()},
            "topic_count": len(by_topic_rules),
        },
        "error_eval_set_gt": summarize(eval_rows),
        "matched_errors_e2e": summarize(matched_rows),
        "notes": {
            "eval_error_type": "Heuristic keyword classification on GT error_text (concept/logic/calculation/modeling/units).",
            "eval_domain": "Sample-level domain from verifier-routed topic in error_symbolic_audit.json.",
            "matched_error_type": "error_type of the matched prediction rule from unified catalog.",
            "matched_domain": "domain of the matched prediction rule from unified catalog.",
        },
    }
    path = out_dir / "distribution_stats.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="catalogs/legacy/unified_rule_library_v2_distilled300_20260503.json",
    )
    parser.add_argument(
        "--eval-dataset",
        default="results/e2e_no_metadata_enhance_30b_error_20260519_145935/error_eval_dataset_100.json",
    )
    parser.add_argument(
        "--e2e-dir",
        default="results/e2e_no_metadata_enhance_30b_error_20260519_145935",
    )
    parser.add_argument(
        "--output-dir",
        default="results/e2e_no_metadata_enhance_30b_error_20260519_145935/figures",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    catalog_path = repo / args.catalog
    eval_path = repo / args.eval_dataset
    e2e_dir = repo / args.e2e_dir
    out_dir = repo / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    catalog = _load_json(catalog_path)
    dataset = _load_json(eval_path)
    metrics = _load_json(e2e_dir / "error_metrics.json")
    audit = _load_json(e2e_dir / "error_symbolic_audit.json")

    topic_to_domain = build_topic_to_domain(catalog)
    rule_index = build_rule_index(catalog)
    sample_domain = {
        str(row.get("id") or ""): topic_to_domain.get(str(row.get("topic") or ""), "Unknown")
        for row in audit
        if isinstance(row, dict)
    }

    by_domain_rules, by_topic_rules = count_rules_by_domain_topic(catalog)
    eval_rows = iter_gt_errors_with_domain(dataset, sample_domain)
    matched_rows = collect_matched_errors(metrics, rule_index, sample_domain, use_rule_type=True)

    plot_rule_library(by_domain_rules, by_topic_rules, out_dir)
    plot_domain_type_heatmap(
        eval_rows,
        "Eval set: locatable GT errors (domain & type)",
        out_dir / "02_error_eval_domain_type_heatmap.png",
    )
    plot_stacked_domain_bars(
        eval_rows,
        "Error-level eval set: error-type mix by domain",
        out_dir / "02_error_eval_domain_type_stacked.png",
    )
    plot_domain_type_heatmap(
        matched_rows,
        "E2E matched errors (domain & type)",
        out_dir / "03_e2e_matched_domain_type_heatmap.png",
    )
    plot_stacked_domain_bars(
        matched_rows,
        "E2E matched errors: type mix by domain",
        out_dir / "03_e2e_matched_domain_type_stacked.png",
    )
    write_stats_json(out_dir, by_domain_rules, by_topic_rules, eval_rows, matched_rows)


if __name__ == "__main__":
    main()
