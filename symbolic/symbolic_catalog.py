from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from symbolic.match_utils import symbol_match_report


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
        self._cache_data: Optional[Dict[str, Any]] = None
        self._cache_mtime: Optional[float] = None
        self._topic_index: Dict[tuple[str, str], List[Dict[str, Any]]] = {}

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            self._cache_data = {"domains": []}
            self._cache_mtime = None
            self._topic_index = {}
            return self._cache_data

        try:
            mtime = self.path.stat().st_mtime
            if self._cache_data is not None and self._cache_mtime == mtime:
                return self._cache_data

            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {"domains": []}
        except Exception:
            data = {"domains": []}
            mtime = None

        self._cache_data = data
        self._cache_mtime = mtime
        self._rebuild_topic_index(data)
        return data

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        self._cache_data = data
        self._cache_mtime = self.path.stat().st_mtime
        self._rebuild_topic_index(data)

    def _rebuild_topic_index(self, data: Dict[str, Any]) -> None:
        index: Dict[tuple[str, str], List[Dict[str, Any]]] = {}
        for domain in data.get("domains", []) or []:
            domain_name = str(domain.get("name") or "")
            for topic in domain.get("topics", []) or []:
                topic_name = str(topic.get("name") or "")
                checks = list(topic.get("checks", []) or [])
                index[(domain_name, topic_name)] = checks
        self._topic_index = index

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

    def find_applicable(
        self,
        domain: str,
        topic: str,
        diagnostic: Dict[str, Any],
        *,
        allow_fallback: bool = False,
        require_symbols_gate: bool = True,
    ) -> List[SymbolicCheckSpec]:
        """Find matching checks in (domain, topic).

        By default this is conservative:
        - It does NOT fall back across topics/domains unless allow_fallback=True.
        - It can (optionally) gate candidates by required_symbols in spec.params, to avoid obvious cross-topic misuse.
        """
        data = self.load()
        rule_id = (diagnostic or {}).get("rule")

        # Build a small text window to match against.
        # We intentionally keep this local to the diagnostic (not the whole sample) to reduce accidental matches.
        evidence = (diagnostic or {}).get("evidence")
        quote = ""
        if isinstance(evidence, dict):
            quote = str(evidence.get("quote") or "")
        msg_raw = str((diagnostic or {}).get("message") or "")
        sym_raw = str((diagnostic or {}).get("symbol") or "")
        haystack = "\n".join([msg_raw, quote, sym_raw]).strip()
        haystack_lower = haystack.lower()

        def _has_required_symbols(c: Dict[str, Any]) -> bool:
            if not require_symbols_gate:
                return True
            params = c.get("params") if isinstance(c.get("params"), dict) else {}
            req = params.get("required_symbols") or []
            if not req:
                return True
            min_ratio = float(params.get("required_symbol_min_ratio", 0.5))
            report = symbol_match_report(haystack, [str(x) for x in req], min_ratio=min_ratio)
            return bool(report.get("ok"))

        def score_check(c: Dict[str, Any]) -> int:
            if not _has_required_symbols(c):
                return 0
            match_rule_ids = [str(x) for x in (c.get("match_rule_ids") or []) if x]
            if rule_id and match_rule_ids and str(rule_id) not in match_rule_ids:
                return 0
            score = 0
            if rule_id and str(rule_id) in match_rule_ids:
                score += 10
            for kw in (c.get("match_keywords") or []):
                if kw and str(kw).lower() in haystack_lower:
                    score += 2
            if not match_rule_ids and score < 4:
                return 0
            return score

        def iter_checks(preferred_only: bool) -> List[Dict[str, Any]]:
            if preferred_only:
                return list(self._topic_index.get((domain, topic), []))

            out: List[Dict[str, Any]] = []
            for checks in self._topic_index.values():
                out.extend(checks)
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
        if allow_fallback:
            return select(others)
        return []
