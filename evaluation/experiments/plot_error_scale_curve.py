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
            try:
                size = int(float(row.get("expansion_size") or 0))
            except ValueError:
                continue
            if size <= 0:
                continue

            def _f(key: str) -> float:
                try:
                    return float(row.get(key) or 0.0)
                except ValueError:
                    return 0.0

            rows.append(
                {
                    "expansion_size": float(size),
                    "precision": _f("precision"),
                    "recall": _f("recall"),
                    "f1": _f("f1"),
                    "sample_trigger_rate": _f("sample_trigger_rate"),
                    "location_hit_ratio": _f("location_hit_ratio"),
                    "catalog_rules": _f("catalog_rules"),
                }
            )
    rows.sort(key=lambda x: x["expansion_size"])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot error-level scale curve from aggregated CSV.")
    parser.add_argument("--input-csv", type=str, default="results/scale_curve_error/curve_metrics.csv")
    parser.add_argument("--output", type=str, default="results/scale_curve_error/error_scale_curve.png")
    parser.add_argument("--title", type=str, default="Error-Level Metrics vs Rule Library Expansion Size (Local 30B)")
    args = parser.parse_args()

    csv_path = Path(args.input_csv)
    out_path = Path(args.output)
    rows = _load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No points found in: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib required: pip install matplotlib") from exc

    x = [int(p["expansion_size"]) for p in rows]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(x, [p["recall"] for p in rows], marker="o", linewidth=2, label="Recall")
    axes[0].plot(x, [p["precision"] for p in rows], marker="o", linewidth=2, label="Precision")
    axes[0].plot(x, [p["f1"] for p in rows], marker="o", linewidth=2, label="F1")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(loc="best")

    axes[1].plot(x, [p["sample_trigger_rate"] for p in rows], marker="o", linewidth=2, label="Sample Trigger Rate")
    axes[1].plot(x, [p["location_hit_ratio"] for p in rows], marker="o", linewidth=2, label="Location Hit Ratio")
    ax2 = axes[1].twinx()
    ax2.plot(x, [p["catalog_rules"] for p in rows], marker="s", linewidth=1.5, color="gray", alpha=0.7, label="Catalog Rules")
    axes[1].set_xlabel("Expansion Sample Count")
    axes[1].set_ylabel("Ratio")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[1].legend(loc="upper left")
    ax2.set_ylabel("Rule Count")
    ax2.legend(loc="upper right")

    fig.suptitle(args.title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    print({"input_csv": str(csv_path), "output": str(out_path), "points": len(rows)})


if __name__ == "__main__":
    main()
