from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Dict


def _load_samples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list at {path}, got {type(data).__name__}")
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise SystemExit(f"Entry #{idx} is not an object: {type(item).__name__}")
    return data


def _renumber(samples: List[Dict[str, Any]], start: int, field: str) -> None:
    current = start
    for sample in samples:
        sample[field] = str(current)
        current += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Renumber entries in evaluation_input-style JSON so that the id field "
            "is a contiguous natural number sequence."
        )
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        default="data/evaluation_input.json",
        help="Path to the JSON file to rewrite (default: data/evaluation_input.json).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional output path. Defaults to in-place overwrite of the input file.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1,
        help="Starting integer for the new IDs (default: 1).",
    )
    parser.add_argument(
        "--field",
        type=str,
        default="id",
        help="JSON field to overwrite with sequential numbers (default: id).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute the new IDs but do not write any files.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    samples = _load_samples(input_path)
    _renumber(samples, start=args.start, field=args.field)

    if args.dry_run:
        print(
            f"Renumbered {len(samples)} entries (start={args.start}, field={args.field}) in dry-run mode."
        )
        return

    output_path = Path(args.output) if args.output else input_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(samples, fp, ensure_ascii=False, indent=2)
    print(
        f"Renumbered {len(samples)} entries written to {output_path} (start={args.start}, field={args.field})."
    )


if __name__ == "__main__":
    main()
