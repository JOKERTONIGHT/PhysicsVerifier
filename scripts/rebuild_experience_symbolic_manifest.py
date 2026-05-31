#!/usr/bin/env python3
"""Rebuild experience-code manifest JSON from a generated checks module."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def _rebuild(module_name: str) -> Dict[str, Any]:
    mod = importlib.import_module(module_name)
    registry = getattr(mod, "EXPERIENCE_CHECK_REGISTRY", None)
    if not isinstance(registry, list):
        raise SystemExit(f"{module_name} has no EXPERIENCE_CHECK_REGISTRY list")

    checks: List[Dict[str, Any]] = []
    skipped = 0
    for item in registry:
        if not isinstance(item, dict):
            skipped += 1
            continue
        rule_id = str(item.get("rule_id") or "").strip()
        fn = item.get("function")
        if not rule_id or not callable(fn):
            skipped += 1
            continue
        func_name = getattr(fn, "__name__", None)
        if not func_name:
            skipped += 1
            continue
        checks.append(
            {
                "rule_id": rule_id,
                "domain": str(item.get("domain") or "Unknown"),
                "topic": str(item.get("topic") or "Unknown"),
                "title": str(item.get("title") or rule_id),
                "path": item.get("path") if isinstance(item.get("path"), dict) else {},
                "symbolic_hint": item.get("symbolic_hint")
                if isinstance(item.get("symbolic_hint"), dict)
                else {},
                "function_name": func_name,
                "source": "rebuilt_from_module_registry",
            }
        )

    return {
        "summary": {
            "rebuilt_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "module": module_name,
            "registry_entries": len(registry),
            "checks_written": len(checks),
            "skipped": skipped,
        },
        "checks": checks,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        default="symbolic.generated_experience_checks_v2_unified",
        help="Python module containing EXPERIENCE_CHECK_REGISTRY",
    )
    parser.add_argument(
        "--output",
        default=str(root / "results/experience_symbolic_program_manifest_v2_unified.json"),
    )
    args = parser.parse_args()

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    manifest = _rebuild(args.module)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = manifest["summary"]
    print(f"Wrote {out}")
    print(
        f"  checks={summary['checks_written']} "
        f"registry={summary['registry_entries']} skipped={summary['skipped']}"
    )


if __name__ == "__main__":
    main()
