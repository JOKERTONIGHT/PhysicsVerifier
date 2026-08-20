from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_rule_coarsening import audit_rule_coarsening
from scripts.build_unified_catalog import build_unified_catalog
from scripts.generate_cluster_proposals import add_catalog_fallback_proposals
from scripts.refine_cluster_blueprints import (
    build_generated_blueprints_from_refined_proposals,
)
from scripts.validate_unified_catalog_structure import validate_catalog_structure


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


def _topic_rule_ids(catalog: Dict[str, Any]) -> Dict[tuple[str, str], set[str]]:
    return {
        (str(domain.get("name") or ""), str(topic.get("name") or "")): {
            str(rule.get("rule_id") or "")
            for rule in topic.get("rules", []) or []
            if str(rule.get("rule_id") or "")
        }
        for domain in catalog.get("domains", []) or []
        for topic in domain.get("topics", []) or []
    }


def finalize_incremental_update(
    *,
    workspace: Path,
    base_catalog_path: Path,
    knowledge_path: Path = Path("catalogs/rules_catalog_top_down.json"),
    tagged_path: Path = Path("catalogs/rules_300_tagged.json"),
    seed_blueprints_path: Path = Path("catalogs/scenario_cluster_blueprints.json"),
) -> Dict[str, Any]:
    manifest = _load_json(workspace / "incremental_manifest.json")
    precluster_catalog = _load_json(workspace / "catalog_precluster.json")
    proposals = add_catalog_fallback_proposals(
        _load_json(workspace / "cluster_proposals.json"),
        precluster_catalog,
    )
    rule_index = {
        str(rule.get("rule_id") or ""): rule
        for domain in precluster_catalog.get("domains", []) or []
        for topic in domain.get("topics", []) or []
        for rule in topic.get("rules", []) or []
        if str(rule.get("rule_id") or "")
    }
    generated_blueprints = build_generated_blueprints_from_refined_proposals(
        proposals,
        rule_index=rule_index,
    )
    generated_blueprints_path = workspace / "cluster_blueprints_generated.json"
    _write_json(generated_blueprints_path, generated_blueprints)

    final_catalog = build_unified_catalog(
        knowledge_path=knowledge_path,
        distilled_path=workspace
        / "semantic_experience_generalized_for_cluster.json",
        tagged_path=tagged_path,
        scenario_cluster_blueprints_paths=[
            seed_blueprints_path,
            generated_blueprints_path,
        ],
    )
    final_catalog_path = workspace / "rules_unified_incremental.json"
    _write_json(final_catalog_path, final_catalog)

    structure = validate_catalog_structure(final_catalog)
    coarsening = audit_rule_coarsening(
        candidates=_load_json(
            workspace / "semantic_experience_distilled_for_cluster.json"
        ),
        generalized=_load_json(
            workspace / "semantic_experience_generalized.json"
        ),
        formal=_load_json(
            workspace / "semantic_experience_generalized_for_cluster.json"
        ),
        catalog=final_catalog,
    )
    base_catalog = _load_json(base_catalog_path)
    base_topics = _topic_rule_ids(base_catalog)
    final_topics = _topic_rule_ids(final_catalog)
    affected_topics = {
        (str(item.get("domain") or ""), str(item.get("topic") or ""))
        for item in manifest.get("affected_topics", []) or []
        if isinstance(item, dict)
    }
    changed_topics = {
        key
        for key in set(base_topics) | set(final_topics)
        if base_topics.get(key, set()) != final_topics.get(key, set())
    }
    unexpected_changed_topics = sorted(changed_topics - affected_topics)
    added_rule_ids = sorted(
        set().union(*final_topics.values()) - set().union(*base_topics.values())
    )
    removed_rule_ids = sorted(
        set().union(*base_topics.values()) - set().union(*final_topics.values())
    )
    identity_stable = not removed_rule_ids
    ready_for_retrieval = bool(
        structure.get("valid")
        and coarsening.get("complete")
        and not unexpected_changed_topics
        and identity_stable
    )

    report = {
        "ready_for_retrieval_evaluation": ready_for_retrieval,
        "promotion_ready": False,
        "promotion_blocker": (
            (
                "Incremental update removed existing Rule IDs; preserve the current "
                "formal catalog before retrieval evaluation."
            )
            if not identity_stable
            else (
                "Review added formal rules and pass an independent full-verifier "
                "regression before replacing the current catalog."
            )
        ),
        "outputs": {
            "catalog": str(final_catalog_path),
            "generated_blueprints": str(generated_blueprints_path),
        },
        "structure": structure,
        "coarsening": coarsening,
        "change_scope": {
            "declared_affected_topics": [
                {"domain": domain, "topic": topic}
                for domain, topic in sorted(affected_topics)
            ],
            "changed_topics": [
                {"domain": domain, "topic": topic}
                for domain, topic in sorted(changed_topics)
            ],
            "unexpected_changed_topics": [
                {"domain": domain, "topic": topic}
                for domain, topic in unexpected_changed_topics
            ],
            "identity_stable": identity_stable,
            "base_rule_ids_preserved": identity_stable,
            "added_rule_ids": added_rule_ids,
            "removed_rule_ids": removed_rule_ids,
        },
    }
    _write_json(workspace / "incremental_validation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate an isolated incremental unified-rule catalog."
    )
    parser.add_argument(
        "--workspace",
        default="results/unified_rules_incremental",
    )
    parser.add_argument("--base-catalog", default="catalogs/rules_unified_3000.json")
    parser.add_argument("--knowledge", default="catalogs/rules_catalog_top_down.json")
    parser.add_argument("--tagged", default="catalogs/rules_300_tagged.json")
    parser.add_argument(
        "--seed-blueprints",
        default="catalogs/scenario_cluster_blueprints.json",
    )
    args = parser.parse_args()

    report = finalize_incremental_update(
        workspace=Path(args.workspace),
        base_catalog_path=Path(args.base_catalog),
        knowledge_path=Path(args.knowledge),
        tagged_path=Path(args.tagged),
        seed_blueprints_path=Path(args.seed_blueprints),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready_for_retrieval_evaluation"] else 1)


if __name__ == "__main__":
    main()
