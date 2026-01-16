from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SymbolicCheckSpec:
    """Data-only symbolic check spec (safe, executable via primitives)."""

    spec_id: str
    title: str
    description: str
    primitive: str
    params: Dict[str, Any]

    # Matching metadata
    match_rule_ids: Optional[List[str]] = None
    match_keywords: Optional[List[str]] = None


class SymbolicCatalog:
    """Maintains symbolic checks grouped by domain/topic, similar to rules catalog."""

    def __init__(self, path: str = "catalogs/symbolic_catalog.json"):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"domains": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"domains": []}

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _get_or_create_domain(self, data: Dict[str, Any], domain_name: str) -> Dict[str, Any]:
        domains = data.setdefault("domains", [])
        for d in domains:
            if d.get("name") == domain_name:
                return d
        nd = {"name": domain_name, "topics": []}
        domains.append(nd)
        return nd

    def _get_or_create_topic(self, domain_obj: Dict[str, Any], topic_name: str) -> Dict[str, Any]:
        topics = domain_obj.setdefault("topics", [])
        for t in topics:
            if t.get("name") == topic_name:
                return t
        nt = {"name": topic_name, "checks": []}
        topics.append(nt)
        return nt

    def upsert_check(self, domain: str, topic: str, spec: SymbolicCheckSpec) -> None:
        data = self.load()
        dom = self._get_or_create_domain(data, domain)
        top = self._get_or_create_topic(dom, topic)
        checks: List[Dict[str, Any]] = top.setdefault("checks", [])

        payload = {
            "spec_id": spec.spec_id,
            "title": spec.title,
            "description": spec.description,
            "primitive": spec.primitive,
            "params": spec.params,
            "match_rule_ids": spec.match_rule_ids or [],
            "match_keywords": spec.match_keywords or [],
        }

        replaced = False
        for i, c in enumerate(checks):
            if c.get("spec_id") == spec.spec_id:
                checks[i] = payload
                replaced = True
                break
        if not replaced:
            checks.append(payload)
        self.save(data)

    def find_applicable(self, domain: str, topic: str, diagnostic: Dict[str, Any]) -> List[SymbolicCheckSpec]:
        """Find matching checks in (domain, topic) first, then fall back to other topics/domains."""
        data = self.load()
        rule_id = (diagnostic or {}).get("rule")
        msg = str((diagnostic or {}).get("message") or "").lower()

        def score_check(c: Dict[str, Any]) -> int:
            score = 0
            if rule_id and rule_id in (c.get("match_rule_ids") or []):
                score += 10
            for kw in (c.get("match_keywords") or []):
                if kw and str(kw).lower() in msg:
                    score += 2
            return score

        def iter_checks(preferred_only: bool) -> List[Dict[str, Any]]:
            out: List[Dict[str, Any]] = []
            for d in data.get("domains", []) or []:
                for t in d.get("topics", []) or []:
                    if preferred_only and (d.get("name") != domain or t.get("name") != topic):
                        continue
                    out.extend(t.get("checks", []) or [])
            return out

        preferred = iter_checks(preferred_only=True)
        others = iter_checks(preferred_only=False)

        def select(candidates: List[Dict[str, Any]]) -> List[SymbolicCheckSpec]:
            scored = [(score_check(c), c) for c in candidates]
            scored = [(s, c) for (s, c) in scored if s > 0]
            scored.sort(key=lambda x: x[0], reverse=True)
            specs: List[SymbolicCheckSpec] = []
            for _, c in scored[:3]:
                try:
                    specs.append(
                        SymbolicCheckSpec(
                            spec_id=str(c.get("spec_id")),
                            title=str(c.get("title")),
                            description=str(c.get("description")),
                            primitive=str(c.get("primitive")),
                            params=dict(c.get("params") or {}),
                            match_rule_ids=list(c.get("match_rule_ids") or []),
                            match_keywords=list(c.get("match_keywords") or []),
                        )
                    )
                except Exception:
                    continue
            return specs

        specs = select(preferred)
        if specs:
            return specs
        return select(others)
