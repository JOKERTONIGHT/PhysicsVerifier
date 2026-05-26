from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.top_down_verifier import TopDownVerifier


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample_id(sample: Dict[str, Any], index: int) -> str:
    return str(sample.get("id") or sample.get("sample_id") or index)


def _compact_result(sample: Dict[str, Any], result: Dict[str, Any], index: int) -> Dict[str, Any]:
    return {
        "sample_id": _sample_id(sample, index),
        "selection_strategy": str(result.get("selection_strategy") or ""),
        "semantic_selection_error": str(result.get("semantic_selection_error") or ""),
        "topic_count": len(result.get("retrieved_topics") or []),
        "cluster_count": len(result.get("retrieved_clusters") or []),
        "rule_count": len(result.get("retrieved_rules") or []),
        "diagnostic_count": len(result.get("diagnostics") or []),
        "retrieved_topics": result.get("retrieved_topics") or [],
        "retrieved_clusters": result.get("retrieved_clusters") or [],
        "retrieved_rules": result.get("retrieved_rules") or [],
    }


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample_count = len(rows)
    strategy_counts = Counter(row["selection_strategy"] for row in rows)
    semantic_errors = [row for row in rows if row["semantic_selection_error"]]
    topic_selected = [row for row in rows if row["topic_count"] > 0]
    cluster_selected = [row for row in rows if row["cluster_count"] > 0]
    rule_selected = [row for row in rows if row["rule_count"] > 0]
    diagnostics = [row for row in rows if row["diagnostic_count"] > 0]
    total_rules = sum(int(row["rule_count"]) for row in rows)
    empty_rule_rows = [row for row in rows if row["rule_count"] == 0]
    high_rule_rows = [row for row in rows if row["rule_count"] > 5]
    broad_topic_rows = [row for row in rows if row["topic_count"] > 2]
    broad_cluster_rows = [row for row in rows if row["cluster_count"] > 3]
    return {
        "sample_count": sample_count,
        "selection_strategy_counts": dict(strategy_counts),
        "semantic_tree_selection_count": int(strategy_counts.get("semantic_tree_selection", 0)),
        "semantic_error_count": len(semantic_errors),
        "topic_selected_count": len(topic_selected),
        "cluster_selected_count": len(cluster_selected),
        "rule_selected_count": len(rule_selected),
        "empty_rule_selection_count": sample_count - len(rule_selected),
        "diagnostic_sample_count": len(diagnostics),
        "total_selected_rules": total_rules,
        "average_selected_rules": round(total_rules / sample_count, 4) if sample_count else 0.0,
        "topic_selection_rate": round(len(topic_selected) / sample_count, 4) if sample_count else 0.0,
        "cluster_selection_rate": round(len(cluster_selected) / sample_count, 4) if sample_count else 0.0,
        "rule_selection_rate": round(len(rule_selected) / sample_count, 4) if sample_count else 0.0,
        "empty_rule_sample_ids": [str(row["sample_id"]) for row in empty_rule_rows],
        "high_rule_selection_sample_ids": [str(row["sample_id"]) for row in high_rule_rows],
        "broad_topic_selection_sample_ids": [str(row["sample_id"]) for row in broad_topic_rows],
        "broad_cluster_selection_sample_ids": [str(row["sample_id"]) for row in broad_cluster_rows],
    }


def evaluate_top_down_runtime(
    *,
    samples_path: Path,
    catalog_path: Path,
    output_path: Path | None = None,
    limit: int = 0,
    verifier_factory: Callable[[], Any] | None = None,
) -> Dict[str, Any]:
    samples_payload = _load_json(samples_path)
    if not isinstance(samples_payload, list):
        raise ValueError(f"Expected list JSON samples: {samples_path}")
    samples = [item for item in samples_payload if isinstance(item, dict)]
    if limit > 0:
        samples = samples[:limit]

    verifier = verifier_factory() if verifier_factory else TopDownVerifier(
        llm_model="gemini-3-flash-preview-thinking",
        unified_rules_path=str(catalog_path),
        log_dir="logs/unified_rules_runtime_eval",
        results_dir="results/unified_rules_runtime_eval",
        enable_agentic_postcheck=False,
        enable_experience_pipeline=False,
    )

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for index, sample in enumerate(samples, start=1):
        try:
            result = verifier.verify(sample)
            if not isinstance(result, dict):
                raise RuntimeError("verifier returned non-object result")
            rows.append(_compact_result(sample, result, index))
            print(f"[runtime-eval] {index}/{len(samples)} sample={_sample_id(sample, index)} rules={rows[-1]['rule_count']}", flush=True)
        except Exception as exc:
            failures.append(
                {
                    "sample_id": _sample_id(sample, index),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[runtime-eval] failed {index}/{len(samples)} sample={_sample_id(sample, index)}: {exc}", flush=True)

    report = {
        "samples": str(samples_path),
        "catalog": str(catalog_path),
        "summary": {
            **_summarize(rows),
            "failure_count": len(failures),
        },
        "rows": rows,
        "failures": failures,
    }
    if output_path:
        _write_json(output_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate top-down runtime retrieval quality for a unified rules catalog.")
    parser.add_argument("--samples", default="data/evaluation_sample_debug_30.json")
    parser.add_argument("--catalog", default="catalogs/rules_unified_3000.json")
    parser.add_argument("--output", default="results/unified_rules_3000/top_down_runtime_eval.json")
    parser.add_argument("--limit", type=int, default=30)
    args = parser.parse_args()

    report = evaluate_top_down_runtime(
        samples_path=Path(args.samples),
        catalog_path=Path(args.catalog),
        output_path=Path(args.output),
        limit=int(args.limit),
    )
    print(json.dumps({"summary": report["summary"]}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
