#!/usr/bin/env python3
"""Plot recall-focused scale curve."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


def _load_rows(path: Path) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = int(float(row.get("expansion_size") or 0))
            if size <= 0:
                continue

            def _f(key: str) -> float:
                return float(row.get(key) or 0.0)

            rows.append(
                {
                    "expansion_size": float(size),
                    "recall": _f("recall"),
                    "precision": _f("precision"),
                    "f1": _f("f1"),
                    "matched_gt_errors": _f("matched_gt_errors"),
                    "catalog_rules": _f("catalog_rules"),
                }
            )
    rows.sort(key=lambda x: x["expansion_size"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot recall-focused error-level scale curve.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--title", type=str, default="", help="Optional figure title (default: none)")
    parser.add_argument(
        "--plateau-after",
        type=int,
        default=0,
        help="If >0, draw plateau annotation from this expansion size (default: off)",
    )
    parser.add_argument("--show-legend", action="store_true", help="Show legend (default: off)")
    args = parser.parse_args()

    rows = _load_rows(Path(args.input_csv))
    if not rows:
        raise SystemExit(f"No points in {args.input_csv}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib required") from exc

    x = [int(p["expansion_size"]) for p in rows]
    recall = [p["recall"] for p in rows]
    matched = [p["matched_gt_errors"] for p in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(x, recall, marker="o", linewidth=2.5, color="#1f77b4")
    for xi, yi in zip(x, recall):
        ax1.annotate(f"{100 * yi:.1f}%", (xi, yi), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    if args.plateau_after > 0 and args.plateau_after in x:
        ax1.axvline(args.plateau_after, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.8)
        ax1.axhspan(
            min(recall[x.index(args.plateau_after):]),
            max(recall) + 0.01,
            xmin=(x.index(args.plateau_after) / (len(x) - 1) if len(x) > 1 else 0),
            alpha=0.08,
            color="#d62728",
        )

    ax1.set_xlabel("Expansion Sample Count")
    ax1.set_ylabel("Recall")
    ax1.set_ylim(0.0, max(recall) * 1.25 + 0.02)
    ax1.set_xticks(x)
    ax1.grid(True, linestyle="--", alpha=0.35)
    if args.show_legend:
        ax1.legend(loc="lower right")

    ax2 = ax1.twinx()
    ax2.bar([xi - 15 for xi in x], matched, width=30, alpha=0.25, color="#2ca02c")
    ax2.set_ylabel("Matched GT Errors (688 total)")
    ax2.set_ylim(0, max(matched) * 1.4)

    if args.title.strip():
        fig.suptitle(args.title.strip())
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    print({"output": str(out), "points": len(rows)})


if __name__ == "__main__":
    main()
