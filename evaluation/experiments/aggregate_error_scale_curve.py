from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_size(path: str, payload: Dict[str, Any]) -> int:
    for key in ("expansion_size", "checkpoint_size", "expansion_sample_count"):
        val = payload.get(key)
        if isinstance(val, int) and val > 0:
            return val
    m = re.search(r"scale_(\d+)", path)
    if m:
        return int(m.group(1))
    m = re.search(r"ckpt_(\d+)", path)
    if m:
        return int(m.group(1))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate error-level metrics across scale checkpoints.")
    parser.add_argument(
        "--metrics-glob",
        type=str,
        default="results/scale_curve_error/scale_*/error_metrics.json",
    )
    parser.add_argument("--output-csv", type=str, default="results/scale_curve_error/curve_metrics.csv")
    parser.add_argument("--output-json", type=str, default="results/scale_curve_error/curve_metrics.json")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.metrics_glob))
    if not paths:
        raise SystemExit(f"No files matched: {args.metrics_glob}")

    rows: List[Dict[str, Any]] = []
    for p in paths:
        payload = _load_json(Path(p))
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else payload
        ckpt = _extract_size(p, summary)
        trigger = summary.get("sample_trigger_rate")
        if trigger is None:
            trigger = summary.get("sample_trigger_ratio")
        loc_hit = summary.get("location_hit_ratio")
        if loc_hit is None:
            loc_hit = summary.get("sample_location_hit_ratio")
        rows.append(
            {
                "expansion_size": ckpt,
                "recall": float(summary.get("recall") or 0.0),
                "precision": float(summary.get("precision") or 0.0),
                "f1": float(summary.get("f1") or 0.0),
                "sample_trigger_rate": float(trigger or 0.0),
                "location_hit_ratio": float(loc_hit or 0.0),
                "gt_errors": int(summary.get("gt_errors") or summary.get("total_gt_errors") or 0),
                "matched_gt_errors": int(summary.get("matched_gt_errors") or summary.get("tp") or 0),
                "catalog_rules": int(summary.get("catalog_rules") or 0),
                "source_file": p,
            }
        )

    rows.sort(key=lambda x: x["expansion_size"])

    csv_path = Path(args.output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    out_json = {"summary": {"metrics_glob": args.metrics_glob, "count": len(rows)}, "points": rows}
    json_path = Path(args.output_json)
    json_path.write_text(json.dumps(out_json, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out_json["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
