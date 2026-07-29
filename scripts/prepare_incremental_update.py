from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.prepare_incremental_candidates import prepare_incremental_candidates
from scripts.prepare_rules_for_cluster import prepare_rules_for_cluster


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command(parts: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(part).replace("\\", "/")) for part in parts)


def prepare_incremental_update(
    *,
    new_candidates_path: Path,
    workspace: Path,
    current_candidates_path: Path,
    current_generalized_path: Path,
    current_formal_path: Path,
    current_cluster_proposals_path: Path,
    current_catalog_path: Path,
    knowledge_path: Path,
    tagged_path: Path,
    baseline_catalog_path: Path,
    candidate_embedding_cache_path: Path,
    formal_embedding_cache_path: Path,
    reset_seed_outputs: bool = False,
) -> Dict[str, Any]:
    workspace.mkdir(parents=True, exist_ok=True)
    merged_path = workspace / "semantic_experience_distilled.json"
    merge_report_path = workspace / "incremental_merge_report.json"
    candidate_for_cluster_path = (
        workspace / "semantic_experience_distilled_for_cluster.json"
    )
    candidate_catalog_path = workspace / "candidate_catalog.json"
    candidate_report_path = workspace / "precluster_report.json"
    candidate_embedding_input_path = workspace / "rule_embedding_input.json"
    generalized_path = workspace / "semantic_experience_generalized.json"
    cluster_proposals_path = workspace / "cluster_proposals.json"

    merged, merge_report = prepare_incremental_candidates(
        current_payload=_load_json(current_candidates_path),
        new_payload=_load_json(new_candidates_path),
        formal_payload=_load_json(current_formal_path),
    )
    _write_json(merged_path, merged)
    _write_json(merge_report_path, merge_report)

    prepare_rules_for_cluster(
        distilled_input=merged_path,
        knowledge_path=knowledge_path,
        tagged_path=tagged_path,
        baseline_catalog_path=None,
        distilled_output=candidate_for_cluster_path,
        catalog_output=candidate_catalog_path,
        report_output=candidate_report_path,
        embedding_input_output=candidate_embedding_input_path,
        scenario_cluster_blueprints_paths=[],
    )

    for source, target in (
        (current_generalized_path, generalized_path),
        (current_cluster_proposals_path, cluster_proposals_path),
    ):
        if reset_seed_outputs or not target.exists():
            _write_json(target, _load_json(source))

    candidate_clusters_path = workspace / "rule_embedding_clusters.json"
    formal_path = workspace / "semantic_experience_generalized_for_cluster.json"
    precluster_catalog_path = workspace / "catalog_precluster.json"
    formal_report_path = workspace / "formal_precluster_report.json"
    formal_embedding_input_path = workspace / "formal_rule_embedding_input.json"
    formal_clusters_path = workspace / "formal_rule_embedding_clusters.json"
    formal_seed_catalog_path = (
        current_catalog_path
        if current_catalog_path.exists()
        else baseline_catalog_path
    )

    commands: List[Dict[str, Any]] = [
        {
            "step": "candidate_embedding",
            "calls_api": True,
            "command": _command(
                [
                    "python",
                    "scripts/run_rule_embedding_clustering.py",
                    "--input",
                    candidate_embedding_input_path,
                    "--output",
                    candidate_clusters_path,
                    "--cache",
                    candidate_embedding_cache_path,
                    "--embedding-model",
                    "text-embedding-3-large",
                    "--similarity-threshold",
                    "0.74",
                    "--min-cluster-size",
                    "4",
                    "--resume",
                ]
            ),
        },
        {
            "step": "candidate_generalization",
            "calls_api": True,
            "command": _command(
                [
                    "python",
                    "scripts/generalize_experience_candidates.py",
                    "--candidates",
                    candidate_for_cluster_path,
                    "--clusters",
                    candidate_clusters_path,
                    "--output",
                    generalized_path,
                    "--model",
                    "deepseek-v4-flash-nothinking",
                    "--fallback-model",
                    "gemini-2.5-flash-nothinking",
                    "--max-clusters",
                    "0",
                    "--max-candidates-per-batch",
                    "12",
                    "--request-timeout",
                    "120",
                    "--attempts",
                    "2",
                    "--resume",
                    "--continue-on-error",
                ]
            ),
        },
        {
            "step": "prepare_formal_rules",
            "calls_api": False,
            "command": _command(
                [
                    "python",
                    "scripts/prepare_rules_for_cluster.py",
                    "--distilled-input",
                    generalized_path,
                    "--knowledge",
                    knowledge_path,
                    "--tagged",
                    tagged_path,
                    "--baseline-catalog",
                    formal_seed_catalog_path,
                    "--preserve-baseline-rule-ids",
                    "--distilled-output",
                    formal_path,
                    "--catalog-output",
                    precluster_catalog_path,
                    "--report-output",
                    formal_report_path,
                    "--embedding-input-output",
                    formal_embedding_input_path,
                ]
            ),
        },
        {
            "step": "formal_embedding",
            "calls_api": True,
            "command": _command(
                [
                    "python",
                    "scripts/run_rule_embedding_clustering.py",
                    "--input",
                    formal_embedding_input_path,
                    "--output",
                    formal_clusters_path,
                    "--cache",
                    formal_embedding_cache_path,
                    "--embedding-model",
                    "text-embedding-3-large",
                    "--similarity-threshold",
                    "0.72",
                    "--min-cluster-size",
                    "4",
                    "--resume",
                ]
            ),
        },
        {
            "step": "cluster_labeling",
            "calls_api": True,
            "command": _command(
                [
                    "python",
                    "scripts/generate_cluster_proposals.py",
                    "--catalog",
                    precluster_catalog_path,
                    "--embedding-clusters",
                    formal_clusters_path,
                    "--rule-input",
                    formal_embedding_input_path,
                    "--output",
                    cluster_proposals_path,
                    "--model",
                    "deepseek-v4-flash-nothinking",
                    "--max-topics",
                    "0",
                    "--min-rule-count",
                    "1",
                    "--request-timeout",
                    "180",
                    "--resume",
                    "--continue-on-error",
                ]
            ),
        },
        {
            "step": "finalize_and_validate",
            "calls_api": False,
            "command": _command(
                [
                    "python",
                    "scripts/finalize_incremental_update.py",
                    "--workspace",
                    workspace,
                    "--base-catalog",
                    current_catalog_path,
                ]
            ),
        },
    ]
    affected_topics = merge_report.get("affected_topics") or []
    manifest = {
        "status": "prepared" if affected_topics else "no_rebuild_needed",
        "base_catalog": str(current_catalog_path),
        "base_catalog_sha256": _sha256(current_catalog_path),
        "formal_seed_catalog": str(formal_seed_catalog_path),
        "new_candidates": str(new_candidates_path),
        "workspace": str(workspace),
        "affected_topics": affected_topics,
        "merge_summary": merge_report.get("summary") or {},
        "commands": commands if affected_topics else [],
        "promotion_policy": (
            "Never overwrite the current catalog automatically. Existing formal rules "
            "must be preserved; finalize, review added rules, run full verifier regression, "
            "then promote manually."
        ),
    }
    _write_json(workspace / "incremental_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare an isolated incremental unified-rule update and its runbook."
    )
    parser.add_argument("--new-candidates", required=True)
    parser.add_argument(
        "--workspace",
        default="results/unified_rules_incremental",
    )
    parser.add_argument(
        "--current-candidates",
        default="results/unified_rules_3000/semantic_experience_distilled_for_cluster.json",
    )
    parser.add_argument(
        "--current-generalized",
        default="results/unified_rules_3000/semantic_experience_generalized.json",
    )
    parser.add_argument(
        "--current-formal",
        default="results/unified_rules_3000/semantic_experience_generalized_for_cluster.json",
    )
    parser.add_argument(
        "--current-cluster-proposals",
        default="results/unified_rules_3000/cluster_proposals.json",
    )
    parser.add_argument("--current-catalog", default="catalogs/rules_unified_3000.json")
    parser.add_argument("--knowledge", default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--tagged", default="catalogs/rules_300_tagged.json")
    parser.add_argument("--baseline-catalog", default="catalogs/rules_unified.json")
    parser.add_argument(
        "--candidate-embedding-cache",
        default="results/unified_rules_3000/rule_embedding_cache.json",
    )
    parser.add_argument(
        "--formal-embedding-cache",
        default="results/unified_rules_3000/formal_rule_embedding_cache.json",
    )
    parser.add_argument("--reset-seed-outputs", action="store_true")
    args = parser.parse_args()

    manifest = prepare_incremental_update(
        new_candidates_path=Path(args.new_candidates),
        workspace=Path(args.workspace),
        current_candidates_path=Path(args.current_candidates),
        current_generalized_path=Path(args.current_generalized),
        current_formal_path=Path(args.current_formal),
        current_cluster_proposals_path=Path(args.current_cluster_proposals),
        current_catalog_path=Path(args.current_catalog),
        knowledge_path=Path(args.knowledge),
        tagged_path=Path(args.tagged),
        baseline_catalog_path=Path(args.baseline_catalog),
        candidate_embedding_cache_path=Path(args.candidate_embedding_cache),
        formal_embedding_cache_path=Path(args.formal_embedding_cache),
        reset_seed_outputs=bool(args.reset_seed_outputs),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
