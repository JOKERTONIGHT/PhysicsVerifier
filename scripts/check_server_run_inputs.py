from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


REQUIRED_SAMPLE_FIELDS = ("id", "question", "prediction", "answer")
AUXILIARY_MARKERS = ('"auxiliary"', '"node_summary"', '"scene_cues"', '"boundary_cues"', '"explore_cues"')


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_report(sample_path: Path, expected_samples: int) -> Dict[str, Any]:
    if not sample_path.exists():
        return {"exists": False, "count": 0, "expected": expected_samples, "empty_required_field_count": 0, "duplicate_id_count": 0}
    payload = _load_json(sample_path)
    samples = payload if isinstance(payload, list) else []
    ids = [str(item.get("id") or "") for item in samples if isinstance(item, dict)]
    return {
        "exists": True,
        "count": len(samples),
        "expected": expected_samples,
        "empty_required_field_count": sum(
            1
            for item in samples
            if not isinstance(item, dict)
            or any(not str(item.get(field) or "").strip() for field in REQUIRED_SAMPLE_FIELDS)
        ),
        "duplicate_id_count": len(ids) - len(set(ids)),
    }


def _run_script_report(run_script_path: Path) -> Dict[str, Any]:
    if not run_script_path.exists():
        return {"exists": False, "has_auxiliary_schema": False}
    text = run_script_path.read_text(encoding="utf-8")
    return {
        "exists": True,
        "has_auxiliary_schema": all(marker in text for marker in AUXILIARY_MARKERS),
    }


def _rules_catalog_report(rules_catalog_path: Path) -> Dict[str, Any]:
    if not rules_catalog_path.exists():
        return {"exists": False, "domain_count": 0}
    payload = _load_json(rules_catalog_path)
    domains = payload.get("domains") if isinstance(payload, dict) else []
    return {"exists": True, "domain_count": len(domains or [])}


def _unified_catalog_report(unified_catalog_path: Path) -> Dict[str, Any]:
    if not unified_catalog_path.exists():
        return {"exists": False, "schema_profile": ""}
    payload = _load_json(unified_catalog_path)
    metadata = payload.get("metadata") if isinstance(payload, dict) else {}
    return {
        "exists": True,
        "schema_profile": str(metadata.get("schema_profile") or ""),
    }


def check_inputs(
    *,
    sample_path: Path,
    expected_samples: int = 3000,
    run_script_path: Path = Path("scripts/run_semantic_experience.py"),
    rules_catalog_path: Path = Path("catalogs/rules_catalog_top_down.json"),
    unified_catalog_path: Path = Path("catalogs/rules_unified.json"),
    output_path: Path | None = None,
) -> Dict[str, Any]:
    sample = _sample_report(sample_path, expected_samples)
    run_script = _run_script_report(run_script_path)
    rules_catalog = _rules_catalog_report(rules_catalog_path)
    unified_catalog = _unified_catalog_report(unified_catalog_path)

    failures = []
    if not sample["exists"]:
        failures.append("sample_missing")
    if sample["count"] != expected_samples:
        failures.append("sample_count_mismatch")
    if sample["empty_required_field_count"]:
        failures.append("sample_empty_required_fields")
    if sample["duplicate_id_count"]:
        failures.append("sample_duplicate_ids")
    if not run_script["exists"]:
        failures.append("run_script_missing")
    if not run_script["has_auxiliary_schema"]:
        failures.append("run_script_missing_auxiliary_schema")
    if not rules_catalog["exists"]:
        failures.append("rules_catalog_missing")
    if not unified_catalog["exists"]:
        failures.append("unified_catalog_missing")
    if unified_catalog["schema_profile"] != "semantic_navigation_tree_minimal":
        failures.append("unified_catalog_schema_not_minimal")

    report = {
        "ready": not failures,
        "failures": failures,
        "sample": sample,
        "run_script": run_script,
        "rules_catalog": rules_catalog,
        "unified_catalog": unified_catalog,
    }
    if output_path:
        _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check local inputs before running 3000 semantic extraction on server.")
    parser.add_argument("--sample", default="data/evaluation_sample_3000_expansion.json")
    parser.add_argument("--expected-samples", type=int, default=3000)
    parser.add_argument("--run-script", default="scripts/run_semantic_experience.py")
    parser.add_argument("--rules-catalog", default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--unified-catalog", default="catalogs/rules_unified.json")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = check_inputs(
        sample_path=Path(args.sample),
        expected_samples=args.expected_samples,
        run_script_path=Path(args.run_script),
        rules_catalog_path=Path(args.rules_catalog),
        unified_catalog_path=Path(args.unified_catalog),
        output_path=Path(args.output) if args.output else None,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
