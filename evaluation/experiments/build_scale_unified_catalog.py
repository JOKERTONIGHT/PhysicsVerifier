#!/usr/bin/env python3
"""Build a scale-point unified catalog via the full per-scale unified_rules pipeline.

Unlike the previous shortcut (prefix semantic + filtered blueprints from the full
3000 catalog), each scale point now runs:

  subset semantic → prepare-cluster (no baseline seed) → embedding clustering
  → LLM cluster labeling → generated blueprints → catalog rebuild
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_unified_catalog import build_unified_catalog  # noqa: E402
from scripts.generate_cluster_proposals import (  # noqa: E402
    _build_client,
    add_catalog_fallback_proposals,
    generate_cluster_proposals_from_embedding_clusters,
)
from scripts.generate_experience_rules import _build_distilled_library, _resume_done_map  # noqa: E402
from scripts.prepare_rules_for_cluster import prepare_rules_for_cluster  # noqa: E402
from scripts.refine_cluster_blueprints import build_generated_blueprints_from_refined_proposals  # noqa: E402
from scripts.run_rule_embedding_clustering import run_embedding_clustering  # noqa: E402
from scripts.subset_semantic_experience_for_scale import _load_expansion_ids  # noqa: E402
from scripts.validate_cluster_blueprints import validate_blueprints_against_catalog  # noqa: E402


def _rules(payload: Any) -> List[Dict[str, Any]]:
    raw = payload.get("rules") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


@contextmanager
def _temporary_openai_env(*, api_key: str | None, base_url: str | None) -> Iterator[None]:
    backup = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL"),
        "OPENAI_API_BASE": os.environ.get("OPENAI_API_BASE"),
    }
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url.rstrip("/")
        os.environ["OPENAI_API_BASE"] = base_url.rstrip("/")
    try:
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _resolve_api_creds(
    *,
    api_key: str | None,
    base_url: str | None,
    fallback_key_env: str,
    fallback_url_env: str,
) -> tuple[str, str | None]:
    key = (api_key or os.getenv(fallback_key_env) or os.getenv("OPENAI_API_KEY") or "").strip()
    url = (base_url or os.getenv(fallback_url_env) or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or "").strip()
    if not key:
        raise RuntimeError(f"{fallback_key_env} / OPENAI_API_KEY is not configured.")
    return key, url or None


def build_scale_unified_catalog(
    *,
    semantic_input: Path,
    expansion_input: Path,
    expansion_size: int,
    work_dir: Path,
    output: Path,
    knowledge_path: Path,
    tagged_path: Path,
    seed_blueprints_path: Path,
    min_rule_count: int = 1,
    embedding_model: str = "text-embedding-3-large",
    embedding_similarity_threshold: float = 0.74,
    embedding_min_cluster_size: int = 4,
    embedding_api_key: str | None = None,
    embedding_base_url: str | None = None,
    cluster_model: str = "qwen3-30b-a3b-instruct-2507",
    cluster_api_key: str | None = None,
    cluster_base_url: str | None = None,
    cluster_max_topics: int = 0,
    cluster_min_rule_count: int = 4,
    cluster_max_rules_per_cluster: int = 8,
    cluster_request_timeout: float = 180.0,
    skip_embedding: bool = False,
    skip_cluster_proposals: bool = False,
    baseline_catalog_path: Path | None = None,
    reuse_artifacts_dir: Path | None = None,
) -> Dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)

    allowed = _load_expansion_ids(expansion_input, int(expansion_size))
    sem_payload = json.loads(semantic_input.read_text(encoding="utf-8"))
    done = _resume_done_map(sem_payload)
    filtered = [done[sid] for sid in sorted(done.keys()) if sid in allowed]
    filtered.sort(key=lambda row: str(row.get("sample_id") or ""))

    sem_path = work_dir / "semantic_experience.json"
    dist_raw_path = work_dir / "semantic_experience_distilled_raw.json"
    _write_json(sem_path, {"samples": filtered})
    distilled_raw = _build_distilled_library(filtered, min_count=max(1, int(min_rule_count)))
    _write_json(dist_raw_path, distilled_raw)

    distilled_for_cluster = work_dir / "semantic_experience_distilled_for_cluster.json"
    interim_catalog = work_dir / "rules_unified_precluster.json"
    precluster_report = work_dir / "precluster_report.json"
    embedding_input = work_dir / "rule_embedding_input.json"
    embedding_clusters = work_dir / "rule_embedding_clusters.json"
    embedding_cache = work_dir / "rule_embedding_cache.json"
    cluster_proposals_path = work_dir / "cluster_proposals.json"
    generated_blueprints_path = work_dir / "cluster_blueprints_generated.json"
    blueprint_validation_path = work_dir / "cluster_blueprints_validation.json"

    if reuse_artifacts_dir is not None:
        reuse_artifacts_dir = Path(reuse_artifacts_dir)
        for name in (
            "cluster_proposals.json",
            "cluster_blueprints_generated.json",
            "rule_embedding_clusters.json",
            "rule_embedding_cache.json",
        ):
            src = reuse_artifacts_dir / name
            dst = work_dir / name
            if src.exists() and not dst.exists():
                dst.write_bytes(src.read_bytes())
        skip_embedding = True
        skip_cluster_proposals = True

    report = prepare_rules_for_cluster(
        distilled_input=dist_raw_path,
        knowledge_path=knowledge_path,
        tagged_path=tagged_path,
        baseline_catalog_path=baseline_catalog_path,
        distilled_output=distilled_for_cluster,
        catalog_output=interim_catalog,
        report_output=precluster_report,
        embedding_input_output=embedding_input,
        scenario_cluster_blueprints_paths=[seed_blueprints_path],
    )

    normalized = json.loads(distilled_for_cluster.read_text(encoding="utf-8"))
    normalized_rule_count = len(_rules(normalized))

    if not skip_embedding:
        embed_key, embed_url = _resolve_api_creds(
            api_key=embedding_api_key,
            base_url=embedding_base_url,
            fallback_key_env="EMBEDDING_OPENAI_API_KEY",
            fallback_url_env="EMBEDDING_OPENAI_BASE_URL",
        )
        with _temporary_openai_env(api_key=embed_key, base_url=embed_url):
            run_embedding_clustering(
                input_path=embedding_input,
                output_path=embedding_clusters,
                cache_path=embedding_cache,
                embedding_model=embedding_model,
                similarity_threshold=float(embedding_similarity_threshold),
                min_cluster_size=int(embedding_min_cluster_size),
                batch_size=64,
            )
    elif not embedding_clusters.exists():
        raise RuntimeError("skip_embedding=1 but embedding clusters file is missing.")

    cluster_summary: Dict[str, Any] = {"skipped": bool(skip_cluster_proposals)}
    if not skip_cluster_proposals:
        cluster_key, cluster_url = _resolve_api_creds(
            api_key=cluster_api_key,
            base_url=cluster_base_url,
            fallback_key_env="CLUSTER_OPENAI_API_KEY",
            fallback_url_env="CLUSTER_OPENAI_BASE_URL",
        )
        client = _build_client(
            api_key=cluster_key,
            base_url=cluster_url,
            trust_env=False,
            request_timeout=float(cluster_request_timeout),
        )
        with _temporary_openai_env(api_key=cluster_key, base_url=cluster_url):
            proposals = generate_cluster_proposals_from_embedding_clusters(
                embedding_clusters=json.loads(embedding_clusters.read_text(encoding="utf-8")),
                rule_input=json.loads(embedding_input.read_text(encoding="utf-8")),
                client=client,
                model=str(cluster_model),
                temperature=0.0,
                max_topics=int(cluster_max_topics),
                min_rule_count=int(cluster_min_rule_count),
                max_rules_per_cluster=int(cluster_max_rules_per_cluster),
                max_output_tokens=2048,
                output_path=cluster_proposals_path,
                resume=True,
                continue_on_error=True,
            )
        interim = json.loads(interim_catalog.read_text(encoding="utf-8"))
        proposals = add_catalog_fallback_proposals(proposals, interim)
        _write_json(cluster_proposals_path, proposals)
        cluster_summary = {
            "proposal_topics": len(proposals.get("proposals") or []),
            "proposal_failures": len(proposals.get("failures") or []),
            "catalog_fallback_topics": int((proposals.get("metadata") or {}).get("catalog_fallback_topic_count") or 0),
        }
    elif cluster_proposals_path.exists():
        interim = json.loads(interim_catalog.read_text(encoding="utf-8"))
        proposals = add_catalog_fallback_proposals(
            json.loads(cluster_proposals_path.read_text(encoding="utf-8")),
            interim,
        )
        _write_json(cluster_proposals_path, proposals)
        cluster_summary = {
            "skipped": True,
            "incremental_catalog_fallback": True,
            "proposal_topics": len(proposals.get("proposals") or []),
            "catalog_fallback_topics": int((proposals.get("metadata") or {}).get("catalog_fallback_topic_count") or 0),
        }
    else:
        raise RuntimeError(
            "skip_cluster_proposals=1 but cluster proposals missing; pass --reuse-artifacts-dir or pre-populate cluster_proposals.json"
        )

    blueprints = build_generated_blueprints_from_refined_proposals(
        json.loads(cluster_proposals_path.read_text(encoding="utf-8"))
    )
    _write_json(generated_blueprints_path, blueprints)

    interim = json.loads(interim_catalog.read_text(encoding="utf-8"))
    validation = validate_blueprints_against_catalog(interim, blueprints, mode="subset")
    _write_json(blueprint_validation_path, validation)

    output.parent.mkdir(parents=True, exist_ok=True)
    catalog = build_unified_catalog(
        knowledge_path=knowledge_path,
        distilled_path=distilled_for_cluster,
        tagged_path=tagged_path,
        scenario_cluster_blueprints_paths=[seed_blueprints_path, generated_blueprints_path],
    )
    _write_json(output, catalog)

    meta = catalog.get("metadata", {})
    summary = {
        "build_mode": "stable_incremental_from_baseline"
        if baseline_catalog_path or reuse_artifacts_dir
        else "full_per_scale_pipeline",
        "expansion_size": int(expansion_size),
        "semantic_samples": len(filtered),
        "distilled_raw_rules": distilled_raw.get("summary", {}).get("total_distilled_rules", 0),
        "normalized_rules": normalized_rule_count,
        "baseline_catalog": str(baseline_catalog_path) if baseline_catalog_path else "",
        "reuse_artifacts_dir": str(reuse_artifacts_dir) if reuse_artifacts_dir else "",
        "baseline_seed_rules": report.get("normalization", {}).get("baseline_seed_rules", 0),
        "generated_blueprint_topics": len(blueprints),
        "generated_blueprint_clusters": sum(len(items) for items in blueprints.values()),
        "catalog_rules": meta.get("total_executable_rules", 0),
        "catalog_clusters": meta.get("total_scenario_clusters", 0),
        "blueprint_validation_valid": bool(validation.get("valid")),
        "cluster_summary": cluster_summary,
        "output": str(output),
    }
    _write_json(work_dir / "build_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified catalog for one scale checkpoint (full pipeline).")
    parser.add_argument("--semantic-input", type=str, required=True)
    parser.add_argument("--expansion-input", type=str, required=True)
    parser.add_argument("--expansion-size", type=int, required=True)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--knowledge", type=str, default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--tagged", type=str, default="catalogs/rules_300_tagged.json")
    parser.add_argument("--seed-blueprints", type=str, default="catalogs/scenario_cluster_blueprints.json")
    parser.add_argument("--min-rule-count", type=int, default=1)
    parser.add_argument("--embedding-model", type=str, default="text-embedding-3-large")
    parser.add_argument("--embedding-similarity-threshold", type=float, default=0.74)
    parser.add_argument("--embedding-min-cluster-size", type=int, default=4)
    parser.add_argument("--embedding-api-key", type=str, default="")
    parser.add_argument("--embedding-base-url", type=str, default="")
    parser.add_argument("--cluster-model", type=str, default="qwen3-30b-a3b-instruct-2507")
    parser.add_argument("--cluster-api-key", type=str, default="")
    parser.add_argument("--cluster-base-url", type=str, default="")
    parser.add_argument("--cluster-max-topics", type=int, default=0, help="0 means all eligible topics.")
    parser.add_argument("--cluster-min-rule-count", type=int, default=4)
    parser.add_argument("--cluster-max-rules-per-cluster", type=int, default=8)
    parser.add_argument("--cluster-request-timeout", type=float, default=180.0)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-cluster-proposals", action="store_true")
    parser.add_argument(
        "--baseline-catalog",
        type=str,
        default="",
        help="Prior scale catalog to seed rules/clusters (e.g. scale_0900 output).",
    )
    parser.add_argument(
        "--reuse-artifacts-dir",
        type=str,
        default="",
        help="Reuse cluster proposals/blueprints from a prior scale build (e.g. scale_0900/catalog_build).",
    )
    args = parser.parse_args()

    summary = build_scale_unified_catalog(
        semantic_input=Path(args.semantic_input),
        expansion_input=Path(args.expansion_input),
        expansion_size=int(args.expansion_size),
        work_dir=Path(args.work_dir),
        output=Path(args.output),
        knowledge_path=Path(args.knowledge),
        tagged_path=Path(args.tagged),
        seed_blueprints_path=Path(args.seed_blueprints),
        min_rule_count=int(args.min_rule_count),
        embedding_model=str(args.embedding_model),
        embedding_similarity_threshold=float(args.embedding_similarity_threshold),
        embedding_min_cluster_size=int(args.embedding_min_cluster_size),
        embedding_api_key=str(args.embedding_api_key or "") or None,
        embedding_base_url=str(args.embedding_base_url or "") or None,
        cluster_model=str(args.cluster_model),
        cluster_api_key=str(args.cluster_api_key or "") or None,
        cluster_base_url=str(args.cluster_base_url or "") or None,
        cluster_max_topics=int(args.cluster_max_topics),
        cluster_min_rule_count=int(args.cluster_min_rule_count),
        cluster_max_rules_per_cluster=int(args.cluster_max_rules_per_cluster),
        cluster_request_timeout=float(args.cluster_request_timeout),
        skip_embedding=bool(args.skip_embedding),
        skip_cluster_proposals=bool(args.skip_cluster_proposals),
        baseline_catalog_path=Path(args.baseline_catalog) if args.baseline_catalog else None,
        reuse_artifacts_dir=Path(args.reuse_artifacts_dir) if args.reuse_artifacts_dir else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
