#!/usr/bin/env python3
"""Apply targeted rule patches for scale_1500 based on failure analysis."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, List


TARGETED_RULE_SPECS: List[Dict[str, Any]] = [
    {
        "id": "targeted_em_flux_closed_surface",
        "title": "Magnetic flux through closed surface must be zero",
        "domain_id": "electromagnetism",
        "topic_id": "electromagnetism.electromagnetic_induction_and_faraday_s_law",
        "trigger": "closed surface magnetic flux integral",
        "check_logic": "Reject using total area vector for flux through a closed surface when B·dA should integrate to zero.",
        "precision": {
            "precision_profile": "strict",
            "publishable": True,
            "preconditions": ["closed surface", "magnetic flux", "gauss"],
            "violation_signatures": ["total surface area vector", "flux through closed surface"],
            "negative_conditions": ["electric flux", "open surface only"],
            "evidence_requirements": ["flux", "closed"],
        },
    },
    {
        "id": "targeted_rel_light_travel_delay",
        "title": "Light travel-time delay in moving observer frames",
        "domain_id": "modern_physics",
        "topic_id": "modern_physics.special_relativity_time_dilation_length_contraction",
        "trigger": "pinhole camera light travel time",
        "check_logic": "Flag solutions that ignore finite speed of light when inferring apparent rod length.",
        "precision": {
            "precision_profile": "strict",
            "publishable": True,
            "preconditions": ["light travel", "pinhole", "apparent length"],
            "violation_signatures": ["instantaneous", "neglect light travel", "simultaneous snapshot"],
            "negative_conditions": ["non-relativistic low speed negligible"],
            "evidence_requirements": ["light", "travel"],
        },
    },
    {
        "id": "targeted_binary_energy_loss",
        "title": "Binary gravitational slingshot energy loss rate",
        "domain_id": "mechanics",
        "topic_id": "mechanics.gravitation_and_kepler_s_laws",
        "trigger": "supermassive black hole binary energy loss",
        "check_logic": "Require explicit slingshot/star encounter energy exchange, not only orbital decay from GW.",
        "precision": {
            "precision_profile": "strict",
            "publishable": True,
            "preconditions": ["binary", "slingshot", "energy loss"],
            "violation_signatures": ["gravitational wave only", "ignore stellar encounters"],
            "negative_conditions": ["single body"],
            "evidence_requirements": ["energy", "binary"],
        },
    },
    {
        "id": "targeted_heat_exchanger_area",
        "title": "Heat exchanger geometry area scaling",
        "domain_id": "thermodynamics_statistical_physics",
        "topic_id": "thermodynamics_statistical_physics.heat_transfer_conduction_convection_radiation",
        "trigger": "counter-flow heat exchanger plate area",
        "check_logic": "Check heat flux uses plate area and conductance with correct thickness scaling.",
        "precision": {
            "precision_profile": "strict",
            "publishable": True,
            "preconditions": ["heat exchanger", "thermal conductance", "plate"],
            "violation_signatures": ["wrong area", "missing thickness", "incorrect conductance units"],
            "negative_conditions": ["adiabatic boundary only"],
            "evidence_requirements": ["heat", "conductance"],
        },
    },
    {
        "id": "targeted_optics_evanescent_decay",
        "title": "Evanescent wave decay direction component",
        "domain_id": "optics",
        "topic_id": "optics.total_internal_reflection",
        "trigger": "total internal reflection evanescent decay",
        "check_logic": "Require normal (cos) components for z-directed decay, not sin components.",
        "precision": {
            "precision_profile": "strict",
            "publishable": True,
            "preconditions": ["evanescent", "total internal reflection", "decay constant"],
            "violation_signatures": ["sin theta", "z-component sin"],
            "negative_conditions": ["propagating transmitted wave"],
            "evidence_requirements": ["evanescent", "decay"],
        },
    },
]


def _iter_topic_rules(catalog: Dict[str, Any]):
    for domain in catalog.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        for topic in domain.get("topics") or []:
            if not isinstance(topic, dict):
                continue
            for rule in topic.get("rules") or []:
                if isinstance(rule, dict):
                    yield domain, topic, rule


def _find_rule(catalog: Dict[str, Any], rule_id: str) -> Dict[str, Any] | None:
    for _, _, rule in _iter_topic_rules(catalog):
        rid = str(rule.get("id") or rule.get("rule_id") or "")
        if rid == rule_id:
            return rule
    return None


def _tighten_top_fp_rules(catalog: Dict[str, Any], failure_by_rule: Dict[str, Any], *, top_n: int = 12) -> List[str]:
    tightened: List[str] = []
    for item in (failure_by_rule.get("top_fp_rules") or [])[:top_n]:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("rule_id") or "")
        causes = item.get("causes") if isinstance(item.get("causes"), dict) else {}
        if int(causes.get("rule_too_broad") or 0) <= 0:
            continue
        rule = _find_rule(catalog, rid)
        if not rule:
            continue
        precision = rule.setdefault("precision", {})
        if not isinstance(precision, dict):
            precision = {}
            rule["precision"] = precision
        precision["publishable"] = False
        precision["precision_profile"] = "strict"
        precision.setdefault("negative_conditions", [])
        if "targeted_tightened_broad_fp" not in precision["negative_conditions"]:
            precision["negative_conditions"].append("targeted_tightened_broad_fp")
        tightened.append(rid)
    return tightened


def _attach_targeted_rules(catalog: Dict[str, Any]) -> List[str]:
    added: List[str] = []
    domain_index = {str(d.get("id") or ""): d for d in catalog.get("domains") or [] if isinstance(d, dict)}
    for spec in TARGETED_RULE_SPECS:
        domain = domain_index.get(str(spec.get("domain_id") or ""))
        if not isinstance(domain, dict):
            continue
        topics = domain.setdefault("topics", [])
        topic = None
        for t in topics:
            if isinstance(t, dict) and str(t.get("id") or "") == str(spec.get("topic_id") or ""):
                topic = t
                break
        if topic is None:
            topic = {
                "id": spec.get("topic_id"),
                "name": str(spec.get("topic_id") or "Targeted Topic"),
                "summary": "Targeted high-miss theme rules",
                "rules": [],
                "scenario_clusters": [],
            }
            topics.append(topic)
        rules = topic.setdefault("rules", [])
        rid = str(spec.get("id") or "")
        if any(str(r.get("id") or r.get("rule_id") or "") == rid for r in rules if isinstance(r, dict)):
            continue
        rules.append(
            {
                "id": rid,
                "rule_id": rid,
                "title": spec.get("title"),
                "trigger": spec.get("trigger"),
                "check_logic": spec.get("check_logic"),
                "description": spec.get("check_logic"),
                "precision": spec.get("precision") or {},
                "symbolic_hint": {},
            }
        )
        added.append(rid)
        clusters = topic.setdefault("scenario_clusters", [])
        if clusters and isinstance(clusters[0], dict):
            clusters[0].setdefault("rule_ids", []).append(rid)
        else:
            topic["scenario_clusters"] = [
                {
                    "id": "targeted_rules_cluster",
                    "name": "Targeted Missed Theme Rules",
                    "summary": "Rules added from missed GT theme mining.",
                    "rule_ids": [rid],
                }
            ]
    meta = catalog.setdefault("metadata", {})
    if isinstance(meta, dict):
        meta["targeted_rule_patch"] = True
        meta["total_executable_rules"] = int(meta.get("total_executable_rules") or 0) + len(added)
    return added


def main() -> None:
    parser = argparse.ArgumentParser(description="Build targeted 1500 catalog from failure-by-rule report.")
    parser.add_argument("--base-catalog", required=True)
    parser.add_argument("--failure-by-rule", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    catalog = json.loads(Path(args.base_catalog).read_text(encoding="utf-8"))
    failure_by_rule = json.loads(Path(args.failure_by_rule).read_text(encoding="utf-8"))
    patched = copy.deepcopy(catalog)

    tightened = _tighten_top_fp_rules(patched, failure_by_rule)
    added = _attach_targeted_rules(patched)

    out = {
        "tightened_rule_ids": tightened,
        "added_rule_ids": added,
        "source_catalog": args.base_catalog,
        "failure_by_rule": args.failure_by_rule,
    }
    patched.setdefault("metadata", {})["targeted_patch_report"] = out

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(patched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
