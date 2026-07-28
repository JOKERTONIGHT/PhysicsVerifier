from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_semantic_experience_run import analyze_run
from scripts.analyze_rule_embedding_clusters import analyze_embedding_clusters
from scripts.check_server_run_inputs import check_inputs
from scripts.evaluate_unified_rules_quality import evaluate_catalog_quality
from scripts.generate_cluster_proposals import add_catalog_fallback_proposals
from scripts.prepare_rules_for_cluster import prepare_rules_for_cluster
from scripts.refine_cluster_blueprints import build_generated_blueprints_from_refined_proposals
from scripts.validate_unified_catalog_structure import validate_catalog_structure


DEFAULT_DATASET = "3000"


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _console_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=True, indent=2)


def dataset_paths(dataset: str = DEFAULT_DATASET, *, root: Path | None = None) -> Dict[str, Path]:
    base = root or Path(".")
    result_dir = base / f"results/unified_rules_{dataset}"
    return {
        "result_dir": result_dir,
        "sample": base / f"data/evaluation_sample_{dataset}_expansion.json",
        "semantic": result_dir / "semantic_experience.json",
        "distilled": result_dir / "semantic_experience_distilled.json",
        "distilled_for_cluster": result_dir / "semantic_experience_distilled_for_cluster.json",
        "generalized": result_dir / "semantic_experience_generalized.json",
        "generalized_for_cluster": result_dir / "semantic_experience_generalized_for_cluster.json",
        "candidate_catalog": result_dir / "candidate_catalog.json",
        "rule_embedding_input": result_dir / "rule_embedding_input.json",
        "rule_embedding_clusters": result_dir / "rule_embedding_clusters.json",
        "rule_embedding_cache": result_dir / "rule_embedding_cache.json",
        "rule_embedding_cluster_report": result_dir / "rule_embedding_cluster_report.json",
        "formal_rule_embedding_input": result_dir / "formal_rule_embedding_input.json",
        "formal_rule_embedding_clusters": result_dir / "formal_rule_embedding_clusters.json",
        "formal_rule_embedding_cache": result_dir / "formal_rule_embedding_cache.json",
        "formal_rule_embedding_cluster_report": result_dir / "formal_rule_embedding_cluster_report.json",
        "extraction_report": result_dir / "extraction_report.json",
        "server_preflight": result_dir / "server_preflight.json",
        "precluster_report": result_dir / "precluster_report.json",
        "formal_precluster_report": result_dir / "formal_precluster_report.json",
        "cluster_proposals": result_dir / "cluster_proposals.json",
        "cluster_proposals_refined": result_dir / "cluster_proposals_refined.json",
        "cluster_blueprints_validation": result_dir / "cluster_blueprints_validation.json",
        "catalog_structure_validation": result_dir / "catalog_structure_validation.json",
        "quality_report": result_dir / "rules_unified_quality_report.json",
        "runtime_eval": result_dir / "top_down_runtime_eval.json",
        "catalog": base / f"catalogs/rules_unified_{dataset}.json",
        "generated_blueprints": result_dir / "cluster_blueprints_generated.json",
    }


def build_server_command(
    *,
    dataset: str = DEFAULT_DATASET,
    model: str = "gemini-3-flash-preview-thinking",
    max_rules_per_sample: int = 2,
    min_rule_count: int = 1,
    resume: bool = True,
) -> str:
    paths = dataset_paths(dataset)
    parts = [
        "python",
        "scripts/generate_experience_rules.py",
        "--input",
        str(paths["sample"]).replace("\\", "/"),
        "--output",
        str(paths["semantic"]).replace("\\", "/"),
        "--distilled-output",
        str(paths["distilled"]).replace("\\", "/"),
        "--max-rules-per-sample",
        str(max_rules_per_sample),
        "--min-rule-count",
        str(min_rule_count),
        "--model",
        model,
    ]
    if resume:
        parts.append("--resume")
    return " ".join(shlex.quote(part) for part in parts)


def build_cluster_proposal_command(
    *,
    dataset: str = DEFAULT_DATASET,
    model: str = "gemini-3-flash-preview-thinking",
    max_topics: int = 0,
    min_rule_count: int = 40,
    max_output_tokens: int = 16384,
    request_timeout: int = 180,
) -> str:
    paths = dataset_paths(dataset)
    parts = [
        "python",
        "scripts/generate_cluster_proposals.py",
        "--catalog",
        str(paths["catalog"]).replace("\\", "/"),
        "--embedding-clusters",
        str(paths["formal_rule_embedding_clusters"]).replace("\\", "/"),
        "--rule-input",
        str(paths["formal_rule_embedding_input"]).replace("\\", "/"),
        "--output",
        str(paths["cluster_proposals"]).replace("\\", "/"),
        "--max-topics",
        str(max_topics),
        "--min-rule-count",
        str(min_rule_count),
        "--model",
        model,
        "--max-output-tokens",
        str(max_output_tokens),
        "--request-timeout",
        str(request_timeout),
        "--resume",
        "--continue-on-error",
    ]
    return " ".join(shlex.quote(part) for part in parts)


def build_blueprint_validation_command(*, dataset: str = DEFAULT_DATASET) -> str:
    paths = dataset_paths(dataset)
    parts = [
        "python",
        "scripts/validate_cluster_blueprints.py",
        "--catalog",
        str(paths["catalog"]).replace("\\", "/"),
        "--blueprints",
        str(paths["generated_blueprints"]).replace("\\", "/"),
        "--output",
        str(paths["cluster_blueprints_validation"]).replace("\\", "/"),
        "--mode",
        "full",
        "--fail-on-invalid",
    ]
    return " ".join(shlex.quote(part) for part in parts)


def build_rebuild_catalog_command(*, dataset: str = DEFAULT_DATASET) -> str:
    paths = dataset_paths(dataset)
    parts = [
        "python",
        "scripts/build_unified_catalog.py",
        "--experience-distilled",
        str(paths["generalized_for_cluster"]).replace("\\", "/"),
        "--scenario-cluster-blueprints",
        "catalogs/scenario_cluster_blueprints.json",
        "--scenario-cluster-blueprints",
        str(paths["generated_blueprints"]).replace("\\", "/"),
        "--output",
        str(paths["catalog"]).replace("\\", "/"),
    ]
    return " ".join(shlex.quote(part) for part in parts)


def build_rule_embedding_cluster_command(
    *,
    dataset: str = DEFAULT_DATASET,
    embedding_model: str = "text-embedding-3-large",
    similarity_threshold: float = 0.74,
    min_cluster_size: int = 4,
    formal: bool = False,
) -> str:
    paths = dataset_paths(dataset)
    input_path = paths["formal_rule_embedding_input"] if formal else paths["rule_embedding_input"]
    output_path = paths["formal_rule_embedding_clusters"] if formal else paths["rule_embedding_clusters"]
    cache_path = paths["formal_rule_embedding_cache"] if formal else paths["rule_embedding_cache"]
    parts = [
        "python",
        "scripts/run_rule_embedding_clustering.py",
        "--input",
        str(input_path).replace("\\", "/"),
        "--output",
        str(output_path).replace("\\", "/"),
        "--cache",
        str(cache_path).replace("\\", "/"),
        "--embedding-model",
        embedding_model,
        "--similarity-threshold",
        str(similarity_threshold),
        "--min-cluster-size",
        str(min_cluster_size),
        "--resume",
    ]
    return " ".join(shlex.quote(part) for part in parts)


def build_catalog_structure_validation_command(*, dataset: str = DEFAULT_DATASET) -> str:
    paths = dataset_paths(dataset)
    parts = [
        "python",
        "scripts/validate_unified_catalog_structure.py",
        "--catalog",
        str(paths["catalog"]).replace("\\", "/"),
        "--output",
        str(paths["catalog_structure_validation"]).replace("\\", "/"),
        "--fail-on-invalid",
    ]
    return " ".join(shlex.quote(part) for part in parts)


def build_generalization_command(
    *,
    dataset: str = DEFAULT_DATASET,
    model: str = "qwen3-30b-a3b",
    max_candidates_per_batch: int = 12,
    request_timeout: int = 180,
) -> str:
    paths = dataset_paths(dataset)
    parts = [
        "python",
        "scripts/generalize_experience_candidates.py",
        "--candidates",
        str(paths["distilled_for_cluster"]).replace("\\", "/"),
        "--clusters",
        str(paths["rule_embedding_clusters"]).replace("\\", "/"),
        "--output",
        str(paths["generalized"]).replace("\\", "/"),
        "--model",
        model,
        "--max-clusters",
        "0",
        "--max-candidates-per-batch",
        str(max_candidates_per_batch),
        "--request-timeout",
        str(request_timeout),
        "--resume",
        "--continue-on-error",
    ]
    return " ".join(shlex.quote(part) for part in parts)


def run_preflight(*, dataset: str = DEFAULT_DATASET, expected_samples: int | None = None) -> Dict[str, Any]:
    paths = dataset_paths(dataset)
    report = check_inputs(
        sample_path=paths["sample"],
        expected_samples=expected_samples or int(dataset),
        output_path=paths["server_preflight"],
    )
    return report


def run_analyze_extraction(
    *,
    dataset: str = DEFAULT_DATASET,
    expected_samples: int | None = None,
    strict: bool = False,
) -> Dict[str, Any]:
    paths = dataset_paths(dataset)
    return analyze_run(
        semantic_path=paths["semantic"],
        distilled_path=paths["distilled"],
        expected_samples=expected_samples or int(dataset),
        output_path=paths["extraction_report"],
        strict=strict,
    )


def run_prepare_cluster(
    *,
    dataset: str = DEFAULT_DATASET,
    root: Path | None = None,
    knowledge_path: Path = Path("catalogs/rules_catalog_top_down.json"),
    tagged_path: Path = Path("catalogs/rules_300_tagged.json"),
    baseline_catalog_path: Path | None = Path("catalogs/rules_unified.json"),
    scenario_cluster_blueprints_paths: Sequence[Path] | None = None,
) -> Dict[str, Any]:
    paths = dataset_paths(dataset, root=root)
    def _resolve(path: Path | None) -> Path | None:
        if path is None:
            return None
        if path.is_absolute() or path.exists() or root is None:
            return path
        return root / path

    knowledge = _resolve(knowledge_path)
    tagged = _resolve(tagged_path)
    baseline = (
        _resolve(baseline_catalog_path)
        if baseline_catalog_path
        else None
    )
    return prepare_rules_for_cluster(
        distilled_input=paths["distilled"],
        knowledge_path=knowledge,
        tagged_path=tagged,
        baseline_catalog_path=baseline,
        distilled_output=paths["distilled_for_cluster"],
        catalog_output=paths["catalog"],
        report_output=paths["precluster_report"],
        embedding_input_output=paths["rule_embedding_input"],
        scenario_cluster_blueprints_paths=scenario_cluster_blueprints_paths,
    )


def _resolve_pipeline_path(path: Path | None, root: Path | None) -> Path | None:
    if path is None:
        return None
    if path.is_absolute() or path.exists() or root is None:
        return path
    return root / path


def run_prepare_candidates(
    *,
    dataset: str = DEFAULT_DATASET,
    root: Path | None = None,
    knowledge_path: Path = Path("catalogs/rules_catalog_top_down.json"),
    tagged_path: Path = Path("catalogs/rules_300_tagged.json"),
) -> Dict[str, Any]:
    """Normalize extracted candidates only; do not mix baseline formal rules."""
    paths = dataset_paths(dataset, root=root)
    return prepare_rules_for_cluster(
        distilled_input=paths["distilled"],
        knowledge_path=_resolve_pipeline_path(knowledge_path, root),
        tagged_path=_resolve_pipeline_path(tagged_path, root),
        baseline_catalog_path=None,
        distilled_output=paths["distilled_for_cluster"],
        catalog_output=paths["candidate_catalog"],
        report_output=paths["precluster_report"],
        embedding_input_output=paths["rule_embedding_input"],
        scenario_cluster_blueprints_paths=[],
    )


def run_prepare_generalized(
    *,
    dataset: str = DEFAULT_DATASET,
    root: Path | None = None,
    knowledge_path: Path = Path("catalogs/rules_catalog_top_down.json"),
    tagged_path: Path = Path("catalogs/rules_300_tagged.json"),
    baseline_catalog_path: Path | None = Path("catalogs/rules_unified.json"),
    scenario_cluster_blueprints_paths: Sequence[Path] | None = None,
) -> Dict[str, Any]:
    """Build the formal rule set after generalization and emit fresh embedding input."""
    paths = dataset_paths(dataset, root=root)
    return prepare_rules_for_cluster(
        distilled_input=paths["generalized"],
        knowledge_path=_resolve_pipeline_path(knowledge_path, root),
        tagged_path=_resolve_pipeline_path(tagged_path, root),
        baseline_catalog_path=_resolve_pipeline_path(baseline_catalog_path, root),
        distilled_output=paths["generalized_for_cluster"],
        catalog_output=paths["catalog"],
        report_output=paths["formal_precluster_report"],
        embedding_input_output=paths["formal_rule_embedding_input"],
        scenario_cluster_blueprints_paths=scenario_cluster_blueprints_paths,
    )


def run_analyze_embedding_clusters(
    *,
    dataset: str = DEFAULT_DATASET,
    root: Path | None = None,
    strict: bool = False,
    min_clustered_rule_ratio: float = 0.3,
    formal: bool = False,
) -> Dict[str, Any]:
    paths = dataset_paths(dataset, root=root)
    return analyze_embedding_clusters(
        input_path=(
            paths["formal_rule_embedding_clusters"]
            if formal
            else paths["rule_embedding_clusters"]
        ),
        output_path=(
            paths["formal_rule_embedding_cluster_report"]
            if formal
            else paths["rule_embedding_cluster_report"]
        ),
        rule_input_path=(
            paths["formal_rule_embedding_input"]
            if formal
            else paths["rule_embedding_input"]
        ),
        min_clustered_rule_ratio=min_clustered_rule_ratio,
        strict=strict,
    )


def run_validate_catalog_structure(
    *,
    dataset: str = DEFAULT_DATASET,
    root: Path | None = None,
) -> Dict[str, Any]:
    paths = dataset_paths(dataset, root=root)
    catalog = json.loads(paths["catalog"].read_text(encoding="utf-8"))
    report = validate_catalog_structure(catalog)
    _write_json(paths["catalog_structure_validation"], report)
    return report


def run_build_blueprints(*, dataset: str = DEFAULT_DATASET, root: Path | None = None) -> Dict[str, Any]:
    paths = dataset_paths(dataset, root=root)
    proposals = json.loads(paths["cluster_proposals"].read_text(encoding="utf-8"))
    catalog = json.loads(paths["catalog"].read_text(encoding="utf-8"))
    proposals = add_catalog_fallback_proposals(proposals, catalog)
    blueprints = build_generated_blueprints_from_refined_proposals(proposals)
    _write_json(paths["generated_blueprints"], blueprints)
    return {
        "generated_blueprints": str(paths["generated_blueprints"]),
        "topic_count": len(blueprints),
        "cluster_count": sum(len(items) for items in blueprints.values()),
        "fallback_topic_count": int(
            (proposals.get("metadata") or {}).get("catalog_fallback_topic_count") or 0
        ),
    }


def run_quality_report(
    *,
    dataset: str = DEFAULT_DATASET,
    root: Path | None = None,
    runtime_eval_path: Path | None = None,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    paths = dataset_paths(dataset, root=root)
    return evaluate_catalog_quality(
        catalog_path=paths["catalog"],
        cluster_proposals_path=paths["cluster_proposals"],
        runtime_eval_path=runtime_eval_path or paths["runtime_eval"],
        output_path=output_path or paths["quality_report"],
    )


def quality_report_exit_code(report: Dict[str, Any], *, fail_on_blocking_gates: bool = False) -> int:
    if not fail_on_blocking_gates:
        return 0
    blocking_count = int((report.get("overall") or {}).get("blocking_gate_count") or 0)
    return 1 if blocking_count else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonical unified_rules workflow entrypoint.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    paths_parser = subparsers.add_parser("paths", help="Print canonical paths for a dataset.")
    paths_parser.add_argument("--dataset", default=DEFAULT_DATASET)

    preflight_parser = subparsers.add_parser("preflight", help="Check local inputs before server extraction.")
    preflight_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    preflight_parser.add_argument("--expected-samples", type=int, default=0)

    command_parser = subparsers.add_parser("server-command", help="Print the server extraction command.")
    command_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    command_parser.add_argument("--model", default="gemini-3-flash-preview-thinking")
    command_parser.add_argument("--max-rules-per-sample", type=int, default=2)
    command_parser.add_argument("--min-rule-count", type=int, default=1)
    command_parser.add_argument("--no-resume", action="store_true")

    analyze_parser = subparsers.add_parser("analyze-extraction", help="Analyze semantic extraction outputs.")
    analyze_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    analyze_parser.add_argument("--expected-samples", type=int, default=0)
    analyze_parser.add_argument("--strict", action="store_true")

    prepare_parser = subparsers.add_parser("prepare-cluster", help="Prepare deterministic rule set for cluster completion.")
    prepare_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    prepare_parser.add_argument("--knowledge", default="catalogs/rules_catalog_top_down.json")
    prepare_parser.add_argument("--tagged", default="catalogs/rules_300_tagged.json")
    prepare_parser.add_argument("--baseline-catalog", default="catalogs/rules_unified.json")
    prepare_parser.add_argument("--scenario-cluster-blueprints", action="append", default=None)

    prepare_candidates_parser = subparsers.add_parser(
        "prepare-candidates",
        help="Normalize extracted candidates and create candidate embedding input.",
    )
    prepare_candidates_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    prepare_candidates_parser.add_argument("--knowledge", default="catalogs/rules_catalog_top_down.json")
    prepare_candidates_parser.add_argument("--tagged", default="catalogs/rules_300_tagged.json")

    generalize_parser = subparsers.add_parser(
        "generalize-command",
        help="Print the candidate generalization command; this step calls API.",
    )
    generalize_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    generalize_parser.add_argument("--model", default="qwen3-30b-a3b")
    generalize_parser.add_argument("--max-candidates-per-batch", type=int, default=12)
    generalize_parser.add_argument("--request-timeout", type=int, default=180)

    prepare_generalized_parser = subparsers.add_parser(
        "prepare-generalized",
        help="Prepare generalized formal rules and fresh formal embedding input.",
    )
    prepare_generalized_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    prepare_generalized_parser.add_argument("--knowledge", default="catalogs/rules_catalog_top_down.json")
    prepare_generalized_parser.add_argument("--tagged", default="catalogs/rules_300_tagged.json")
    prepare_generalized_parser.add_argument("--baseline-catalog", default="catalogs/rules_unified.json")
    prepare_generalized_parser.add_argument("--scenario-cluster-blueprints", action="append", default=None)

    cluster_parser = subparsers.add_parser("cluster-command", help="Print the cluster proposal command; this step calls API.")
    cluster_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    cluster_parser.add_argument("--model", default="gemini-3-flash-preview-thinking")
    cluster_parser.add_argument("--max-topics", type=int, default=0)
    cluster_parser.add_argument("--min-rule-count", type=int, default=40)
    cluster_parser.add_argument("--max-output-tokens", type=int, default=16384)
    cluster_parser.add_argument("--request-timeout", type=int, default=180)

    embedding_parser = subparsers.add_parser("embedding-command", help="Print the rule embedding clustering command; this step calls API.")
    embedding_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    embedding_parser.add_argument("--embedding-model", default="text-embedding-3-large")
    embedding_parser.add_argument("--similarity-threshold", type=float, default=0.74)
    embedding_parser.add_argument("--min-cluster-size", type=int, default=4)

    formal_embedding_parser = subparsers.add_parser(
        "formal-embedding-command",
        help="Print formal-rule embedding clustering command; this step calls API.",
    )
    formal_embedding_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    formal_embedding_parser.add_argument("--embedding-model", default="text-embedding-3-large")
    formal_embedding_parser.add_argument("--similarity-threshold", type=float, default=0.74)
    formal_embedding_parser.add_argument("--min-cluster-size", type=int, default=4)

    analyze_embedding_parser = subparsers.add_parser("analyze-embedding", help="Analyze rule embedding clustering output.")
    analyze_embedding_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    analyze_embedding_parser.add_argument("--strict", action="store_true")
    analyze_embedding_parser.add_argument("--min-clustered-rule-ratio", type=float, default=0.3)
    analyze_embedding_parser.add_argument("--formal", action="store_true")

    blueprints_parser = subparsers.add_parser("build-blueprints", help="Build scenario cluster blueprints from labeled cluster proposals.")
    blueprints_parser.add_argument("--dataset", default=DEFAULT_DATASET)

    validate_command_parser = subparsers.add_parser("validate-blueprints-command", help="Print blueprint validation command.")
    validate_command_parser.add_argument("--dataset", default=DEFAULT_DATASET)

    rebuild_command_parser = subparsers.add_parser("rebuild-command", help="Print catalog rebuild command using generated blueprints.")
    rebuild_command_parser.add_argument("--dataset", default=DEFAULT_DATASET)

    validate_catalog_parser = subparsers.add_parser(
        "validate-catalog",
        help="Validate final catalog identities and rule-tree reachability.",
    )
    validate_catalog_parser.add_argument("--dataset", default=DEFAULT_DATASET)

    validate_catalog_command_parser = subparsers.add_parser(
        "validate-catalog-command",
        help="Print final catalog structural validation command.",
    )
    validate_catalog_command_parser.add_argument("--dataset", default=DEFAULT_DATASET)

    quality_parser = subparsers.add_parser("quality-report", help="Evaluate unified rules catalog quality.")
    quality_parser.add_argument("--dataset", default=DEFAULT_DATASET)
    quality_parser.add_argument("--runtime-eval", default="", help="Override runtime eval path, e.g. targeted capped eval.")
    quality_parser.add_argument("--output", default="", help="Override quality report output path.")
    quality_parser.add_argument("--fail-on-blocking-gates", action="store_true", help="Exit non-zero when readiness gates fail.")

    args = parser.parse_args()

    if args.command == "paths":
        print(_console_json({key: str(value) for key, value in dataset_paths(args.dataset).items()}))
    elif args.command == "preflight":
        report = run_preflight(
            dataset=args.dataset,
            expected_samples=args.expected_samples or None,
        )
        print(_console_json(report))
        if not report.get("ready"):
            raise SystemExit(1)
    elif args.command == "server-command":
        print(
            build_server_command(
                dataset=args.dataset,
                model=args.model,
                max_rules_per_sample=args.max_rules_per_sample,
                min_rule_count=args.min_rule_count,
                resume=not bool(args.no_resume),
            )
        )
    elif args.command == "analyze-extraction":
        report = run_analyze_extraction(
            dataset=args.dataset,
            expected_samples=args.expected_samples or None,
            strict=bool(args.strict),
        )
        print(_console_json(report))
    elif args.command == "prepare-cluster":
        report = run_prepare_cluster(
            dataset=args.dataset,
            knowledge_path=Path(args.knowledge),
            tagged_path=Path(args.tagged),
            baseline_catalog_path=Path(args.baseline_catalog) if args.baseline_catalog else None,
            scenario_cluster_blueprints_paths=(
                [Path(item) for item in args.scenario_cluster_blueprints]
                if args.scenario_cluster_blueprints is not None
                else None
            ),
        )
        print(_console_json({"normalization": report["normalization"], "catalog": report["catalog"]}))
    elif args.command == "prepare-candidates":
        report = run_prepare_candidates(
            dataset=args.dataset,
            knowledge_path=Path(args.knowledge),
            tagged_path=Path(args.tagged),
        )
        print(_console_json({"normalization": report["normalization"], "catalog": report["catalog"]}))
    elif args.command == "generalize-command":
        print(
            build_generalization_command(
                dataset=args.dataset,
                model=args.model,
                max_candidates_per_batch=args.max_candidates_per_batch,
                request_timeout=args.request_timeout,
            )
        )
    elif args.command == "prepare-generalized":
        report = run_prepare_generalized(
            dataset=args.dataset,
            knowledge_path=Path(args.knowledge),
            tagged_path=Path(args.tagged),
            baseline_catalog_path=Path(args.baseline_catalog) if args.baseline_catalog else None,
            scenario_cluster_blueprints_paths=(
                [Path(item) for item in args.scenario_cluster_blueprints]
                if args.scenario_cluster_blueprints is not None
                else None
            ),
        )
        print(_console_json({"normalization": report["normalization"], "catalog": report["catalog"]}))
    elif args.command == "cluster-command":
        print(
            build_cluster_proposal_command(
                dataset=args.dataset,
                model=args.model,
                max_topics=args.max_topics,
                min_rule_count=args.min_rule_count,
                max_output_tokens=args.max_output_tokens,
                request_timeout=args.request_timeout,
            )
        )
    elif args.command == "embedding-command":
        print(
            build_rule_embedding_cluster_command(
                dataset=args.dataset,
                embedding_model=args.embedding_model,
                similarity_threshold=float(args.similarity_threshold),
                min_cluster_size=args.min_cluster_size,
            )
        )
    elif args.command == "formal-embedding-command":
        print(
            build_rule_embedding_cluster_command(
                dataset=args.dataset,
                embedding_model=args.embedding_model,
                similarity_threshold=float(args.similarity_threshold),
                min_cluster_size=args.min_cluster_size,
                formal=True,
            )
        )
    elif args.command == "analyze-embedding":
        report = run_analyze_embedding_clusters(
            dataset=args.dataset,
            strict=bool(args.strict),
            min_clustered_rule_ratio=float(args.min_clustered_rule_ratio),
            formal=bool(args.formal),
        )
        print(_console_json({key: value for key, value in report.items() if key != "topics"}))
    elif args.command == "build-blueprints":
        print(_console_json(run_build_blueprints(dataset=args.dataset)))
    elif args.command == "validate-blueprints-command":
        print(build_blueprint_validation_command(dataset=args.dataset))
    elif args.command == "rebuild-command":
        print(build_rebuild_catalog_command(dataset=args.dataset))
    elif args.command == "validate-catalog":
        report = run_validate_catalog_structure(dataset=args.dataset)
        print(_console_json(report))
        if not report["valid"]:
            raise SystemExit(1)
    elif args.command == "validate-catalog-command":
        print(build_catalog_structure_validation_command(dataset=args.dataset))
    elif args.command == "quality-report":
        report = run_quality_report(
            dataset=args.dataset,
            runtime_eval_path=Path(args.runtime_eval) if args.runtime_eval else None,
            output_path=Path(args.output) if args.output else None,
        )
        print(_console_json({
            "overall": report["overall"],
            "schema": report["schema"],
            "cluster_quality": {
                key: value
                for key, value in report["cluster_quality"].items()
                if key != "topics"
            },
            "runtime_eval": report["runtime_eval"],
        }))
        raise SystemExit(quality_report_exit_code(report, fail_on_blocking_gates=bool(args.fail_on_blocking_gates)))
if __name__ == "__main__":
    main()
