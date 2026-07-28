from __future__ import annotations

from dataclasses import dataclass, field
import json
import re
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class SymbolicCheckSpec:
    spec_id: str
    title: str
    description: str
    primitive: str
    params: Dict[str, Any] = field(default_factory=dict)
    match_rule_ids: List[str] = field(default_factory=list)
    match_keywords: List[str] = field(default_factory=list)
    # Populated when loading a catalog with a nested domains/topics layout;
    # used to avoid applying unrelated symbolic primitives across the entire library.
    catalog_domain: str = ""
    catalog_topic: str = ""


def _norm_topic_key(value: str) -> str:
    s = str(value or "").strip().lower()
    if "/" in s:
        s = s.split("/")[-1].strip()
    return " ".join(s.split())


def _topic_soft_match(cat_domain: str, cat_topic: str, run_domain: str, run_topic: str) -> bool:
    cd = _norm_topic_key(cat_domain)
    ct = _norm_topic_key(cat_topic)
    rd = _norm_topic_key(run_domain)
    rt = _norm_topic_key(run_topic)

    # No catalog locality → caller may still use strict keyword bridging only.
    if not cd and not ct:
        return True

    if cd and rd and cd not in rd and rd not in cd:
        return False

    if not ct:
        return True

    ct_tokens = {t for t in re.findall(r"[a-z0-9]{4,}", ct)}
    rt_tokens = {t for t in re.findall(r"[a-z0-9]{4,}", rt)}
    overlap = ct_tokens & rt_tokens
    return ct in rt or rt in ct or ct in "".join(rt.split()) or bool(overlap)


def _diagnostic_match_haystack(diagnostic: Dict[str, Any]) -> str:
    parts: List[str] = []
    msg = diagnostic.get("message")
    parts.append("" if msg is None else str(msg))
    sym = diagnostic.get("symbol")
    parts.append("" if sym is None else str(sym))
    evidence = diagnostic.get("evidence")
    if isinstance(evidence, dict):
        quote = evidence.get("quote")
        parts.append("" if quote is None else str(quote))
    return "\n".join(parts).strip().lower()


class SymbolicCatalog:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._specs: List[SymbolicCheckSpec] = []
        self._load()

    def _parse_spec_row(
        self,
        row: Dict[str, Any],
        *,
        catalog_domain: str = "",
        catalog_topic: str = "",
    ) -> Optional[SymbolicCheckSpec]:
        if not isinstance(row, dict):
            return None
        sid = str(row.get("spec_id") or row.get("id") or "").strip()
        if not sid:
            return None
        return SymbolicCheckSpec(
            spec_id=sid,
            title=str(row.get("title") or ""),
            description=str(row.get("description") or ""),
            primitive=str(row.get("primitive") or ""),
            params=dict(row.get("params") or {}),
            match_rule_ids=[str(x) for x in (row.get("match_rule_ids") or []) if str(x).strip()],
            match_keywords=[str(x) for x in (row.get("match_keywords") or []) if str(x).strip()],
            catalog_domain=str(catalog_domain or ""),
            catalog_topic=str(catalog_topic or ""),
        )

    def _load_flat(self, data: Any) -> List[SymbolicCheckSpec]:
        specs: List[SymbolicCheckSpec] = []
        if isinstance(data, list):
            iterable = data
        elif isinstance(data, dict):
            iterable = data.get("specs") or data.get("items") or []
        else:
            iterable = []
        for row in iterable:
            sp = self._parse_spec_row(row if isinstance(row, dict) else {})
            if sp is not None:
                specs.append(sp)
        return specs

    def _load_nested_domains(self, data: Dict[str, Any]) -> List[SymbolicCheckSpec]:
        specs: List[SymbolicCheckSpec] = []
        domains = data.get("domains")
        if not isinstance(domains, list):
            return specs

        for dom in domains:
            if not isinstance(dom, dict):
                continue
            dname = str(dom.get("name") or "").strip()
            topics = dom.get("topics")
            if not isinstance(topics, list):
                continue
            for top in topics:
                if not isinstance(top, dict):
                    continue
                tname = str(top.get("name") or "").strip()
                checks = top.get("checks")
                if not isinstance(checks, list):
                    continue
                for row in checks:
                    sp = self._parse_spec_row(row if isinstance(row, dict) else {}, catalog_domain=dname, catalog_topic=tname)
                    if sp is not None:
                        specs.append(sp)
        return specs

    def _load(self) -> None:
        if not self.path.exists():
            self._specs = []
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            self._specs = []
            return

        specs: List[SymbolicCheckSpec] = []
        if isinstance(data, dict):
            nested = self._load_nested_domains(data)
            specs.extend(nested)

            flat = self._load_flat(data)
            by_id = {s.spec_id: s for s in specs}

            merged: List[SymbolicCheckSpec] = list(specs)
            for s in flat:
                if s.spec_id not in by_id:
                    merged.append(s)
                    by_id[s.spec_id] = s
            specs = merged
        elif isinstance(data, list):
            specs = self._load_flat(data)

        self._specs = specs

    def find_applicable(self, domain: str, topic: str, diagnostic: Dict[str, Any]) -> List[SymbolicCheckSpec]:
        """Return symbolic checks tied to diagnostic rule IDs, otherwise bridge via keywords (+ topic locality).

        Experience / distilled unified rules commonly use opaque ``exp_*`` IDs that do not appear in legacy
        ``match_rule_ids``; for those we conservatively activate catalog entries whose keywords hit the LLM message
        and whose catalog topic/domain overlaps the verifier's routed topic."""
        rid = str((diagnostic or {}).get("rule") or "").strip()
        if not self._specs:
            return []

        direct = [s for s in self._specs if rid and rid in s.match_rule_ids]
        if direct:
            return direct

        exp_like = bool(rid and (rid.startswith("exp_") or rid.startswith("experience::")))
        if not diagnostic or not isinstance(diagnostic, dict):
            return []
        hay = _diagnostic_match_haystack(diagnostic)
        if not hay.strip():
            return []

        scored: List[tuple[float, SymbolicCheckSpec]] = []
        for s in self._specs:
            if not s.match_keywords:
                continue

            locality_ok = _topic_soft_match(s.catalog_domain, s.catalog_topic, domain, topic)
            if not locality_ok:
                continue

            hit_count = sum(1 for kw in s.match_keywords if kw.lower() in hay)
            if hit_count <= 0:
                continue

            kw_long_hit = sum(1 for kw in s.match_keywords if len(kw) >= 22 and kw.lower() in hay)
            penalty = 0.0 if (s.catalog_topic or exp_like) else 1.5
            score = float(hit_count + 2 * kw_long_hit - penalty)

            min_hits = 2
            long_single = kw_long_hit >= 1
            if hit_count >= min_hits or long_single:
                scored.append((score, s))

        scored.sort(key=lambda x: -x[0])
        if not scored:
            return []

        selected: List[SymbolicCheckSpec] = []
        for _sc, spec in scored:
            if len(selected) >= 3:
                break
            selected.append(spec)

        seen_ids: List[str] = []
        finals: List[SymbolicCheckSpec] = []
        for spec in selected:
            if spec.spec_id in seen_ids:
                continue
            seen_ids.append(spec.spec_id)
            finals.append(spec)
        return finals
