from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rule_framework.builder import build_unified_catalog  # noqa: E402
from rule_framework.io import load_json, write_json  # noqa: E402
from rule_framework.llm_enhancer import enhance_catalog  # noqa: E402
from rule_framework.maintenance import add_experience_rules, attach_symbolic_bindings, recluster_catalog, remove_rules  # noqa: E402
from rule_framework.validation import validate_catalog  # noqa: E402


def _load_rules(path: str) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict):
        rules = data.get("rules") or data.get("checks") or []
    else:
        rules = data
    if not isinstance(rules, list):
        raise SystemExit(f"{path} must be a JSON list or an object containing a 'rules' list.")
    return [rule for rule in rules if isinstance(rule, dict)]


def _write_result(output: str | None, payload: Dict[str, Any]) -> None:
    if output:
        write_json(output, payload)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Reusable hierarchical rule-library maintenance CLI.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Build a unified v2 rule catalog from knowledge + experience sources.")
    build.add_argument("--knowledge", type=str, default="catalogs/rules_catalog_top_down.json")
    build.add_argument("--experience-tagged", type=str, default="catalogs/rules_300_tagged.json")
    build.add_argument("--experience-distilled", type=str, default="catalogs/semantic_experience_distilled_300.json")
    build.add_argument(
        "--topic-alias",
        type=str,
        default="",
        help="Optional JSON map of topic_key -> canonical topic_key for distilled rule routing when skeleton keys differ.",
    )
    build.add_argument("--output", "-o", type=str, required=True)

    add = sub.add_parser("add", help="Add new experience rules into an existing catalog.")
    add.add_argument("--catalog", type=str, required=True)
    add.add_argument("--rules", type=str, required=True, help="JSON list or object with a top-level 'rules' list.")
    add.add_argument("--output", "-o", type=str, required=True)

    remove = sub.add_parser("remove", help="Remove rules by rule_id.")
    remove.add_argument("--catalog", type=str, required=True)
    remove.add_argument("--rule-id", action="append", default=[], help="Rule id to remove; can be repeated.")
    remove.add_argument("--rule-ids-file", type=str, default="", help="Optional JSON/text file containing rule ids.")
    remove.add_argument("--output", "-o", type=str, required=True)

    recluster = sub.add_parser("recluster", help="Recompute clusters and rule tree.")
    recluster.add_argument("--catalog", type=str, required=True)
    recluster.add_argument("--domain", type=str, default=None)
    recluster.add_argument("--topic", type=str, default=None)
    recluster.add_argument("--output", "-o", type=str, required=True)

    bind = sub.add_parser("bind-symbolic", help="Attach generated symbolic-code manifest entries to rule leaves.")
    bind.add_argument("--catalog", type=str, required=True)
    bind.add_argument("--manifest", type=str, required=True)
    bind.add_argument("--output", "-o", type=str, required=True)

    validate = sub.add_parser("validate", help="Validate catalog shape and rule_id uniqueness.")
    validate.add_argument("--catalog", type=str, required=True)
    validate.add_argument("--output", "-o", type=str, default="")

    enhance = sub.add_parser(
        "enhance",
        help="Add LLM-generated retrieval signals (match_phrases, discriminative_terms, semantic clusters).",
    )
    enhance.add_argument("--catalog", type=str, required=True, help="Input unified v2 catalog.")
    enhance.add_argument("--output", "-o", type=str, required=True, help="Output path for enhanced catalog.")
    enhance.add_argument("--model", type=str, default="qwen3-30b-a3b-instruct-2507")
    enhance.add_argument("--no-rule-hints", action="store_true", help="Skip per-rule LLM hint generation.")
    enhance.add_argument("--no-topic-hints", action="store_true", help="Skip per-topic LLM hint generation.")
    enhance.add_argument("--no-semantic-clusters", action="store_true", help="Skip LLM semantic clustering.")
    enhance.add_argument("--cluster-min-rules", type=int, default=4,
                         help="Minimum rules per topic to trigger semantic clustering.")
    enhance.add_argument("--rule-batch-size", type=int, default=6,
                         help="Number of rules to process per LLM call (default 6).")
    enhance.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between LLM calls.")
    enhance.add_argument(
        "--refresh-existing",
        action="store_true",
        help="Regenerate existing LLM hints instead of only filling missing coverage.",
    )
    enhance.add_argument("--rule-id", action="append", default=[], help="Only enhance this rule id; can be repeated.")
    enhance.add_argument(
        "--rule-ids-file",
        type=str,
        default="",
        help="Optional JSON/text file containing rule ids to enhance. Keeps API use focused on FP rules.",
    )

    args = parser.parse_args()

    if args.command == "build":
        alias_path = Path(args.topic_alias) if str(args.topic_alias or "").strip() else None
        catalog = build_unified_catalog(
            knowledge_path=Path(args.knowledge),
            distilled_path=Path(args.experience_distilled),
            tagged_path=Path(args.experience_tagged),
            topic_alias_path=alias_path,
        )
        write_json(args.output, catalog)
        print(f"Built catalog: {args.output}")
        return

    if args.command == "add":
        result = add_experience_rules(load_json(args.catalog), _load_rules(args.rules))
        write_json(args.output, result.catalog)
        print(json.dumps({"added": result.changed_rule_ids, "warnings": result.warnings}, ensure_ascii=False, indent=2))
        return

    if args.command == "remove":
        rule_ids = list(args.rule_id or [])
        if args.rule_ids_file:
            raw = Path(args.rule_ids_file).read_text(encoding="utf-8")
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    rule_ids.extend(str(item) for item in parsed)
                elif isinstance(parsed, dict):
                    rule_ids.extend(str(item) for item in parsed.get("rule_ids") or [])
            except json.JSONDecodeError:
                rule_ids.extend(line.strip() for line in raw.splitlines() if line.strip())
        result = remove_rules(load_json(args.catalog), rule_ids)
        write_json(args.output, result.catalog)
        print(json.dumps({"removed": result.changed_rule_ids, "warnings": result.warnings}, ensure_ascii=False, indent=2))
        return

    if args.command == "recluster":
        result = recluster_catalog(load_json(args.catalog), domain=args.domain, topic=args.topic)
        write_json(args.output, result.catalog)
        print(json.dumps({"reclustered_rules": len(result.changed_rule_ids)}, ensure_ascii=False, indent=2))
        return

    if args.command == "bind-symbolic":
        result = attach_symbolic_bindings(load_json(args.catalog), load_json(args.manifest))
        write_json(args.output, result.catalog)
        print(json.dumps({"bound_rules": result.changed_rule_ids}, ensure_ascii=False, indent=2))
        return

    if args.command == "validate":
        result = validate_catalog(load_json(args.catalog)).as_dict()
        _write_result(args.output, result)
        if not result["ok"]:
            raise SystemExit(1)
        return

    if args.command == "enhance":
        catalog = load_json(args.catalog)
        rule_ids_filter = set(str(item).strip() for item in (args.rule_id or []) if str(item).strip())
        if args.rule_ids_file:
            raw = Path(args.rule_ids_file).read_text(encoding="utf-8")
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    rule_ids_filter.update(str(item).strip() for item in parsed if str(item).strip())
                elif isinstance(parsed, dict):
                    rule_ids_filter.update(
                        str(item).strip() for item in (parsed.get("rule_ids") or []) if str(item).strip()
                    )
            except json.JSONDecodeError:
                rule_ids_filter.update(line.strip() for line in raw.splitlines() if line.strip())
        catalog = enhance_catalog(
            catalog,
            model=args.model,
            do_rule_hints=not args.no_rule_hints,
            do_topic_hints=not args.no_topic_hints,
            do_semantic_clusters=not args.no_semantic_clusters,
            cluster_min_rules=args.cluster_min_rules,
            rule_batch_size=args.rule_batch_size,
            sleep_between_calls=args.sleep,
            refresh_existing=args.refresh_existing,
            rule_ids_filter=rule_ids_filter or None,
            verbose=True,
        )
        write_json(args.output, catalog)
        print(f"Enhanced catalog written to: {args.output}")
        return


if __name__ == "__main__":
    main()
