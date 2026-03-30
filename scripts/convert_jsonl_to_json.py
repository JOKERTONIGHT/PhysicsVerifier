from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List


def _load_jsonl(path: Path) -> List[Any]:
    items: List[Any] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON at line {line_no}: {exc}") from exc
    return items


def _dump_json(path: Path, data: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert JSONL file to standard JSON array")
    ap.add_argument(
        "--input",
        default="data/physics_rubric_data_1000.jsonl",
        help="Path to source JSONL file",
    )
    ap.add_argument(
        "--output",
        default="data/physics_rubric_data_1000.json",
        help="Path to target JSON file",
    )
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    data = _load_jsonl(input_path)
    _dump_json(output_path, data)

    print(f"Converted {len(data)} records: {input_path} -> {output_path}")


if __name__ == "__main__":
    main()
