from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SymbolicCheckSpec:
    spec_id: str
    title: str
    description: str
    primitive: str
    params: Dict[str, Any] = field(default_factory=dict)
    match_rule_ids: List[str] = field(default_factory=list)
    match_keywords: List[str] = field(default_factory=list)


class SymbolicCatalog:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self._specs: List[SymbolicCheckSpec] = []
        self._load()

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
        if isinstance(data, list):
            iterable = data
        elif isinstance(data, dict):
            iterable = data.get("specs") or data.get("items") or []
        else:
            iterable = []

        for row in iterable:
            if not isinstance(row, dict):
                continue
            sid = str(row.get("spec_id") or row.get("id") or "").strip()
            if not sid:
                continue
            specs.append(
                SymbolicCheckSpec(
                    spec_id=sid,
                    title=str(row.get("title") or ""),
                    description=str(row.get("description") or ""),
                    primitive=str(row.get("primitive") or ""),
                    params=dict(row.get("params") or {}),
                    match_rule_ids=[str(x) for x in (row.get("match_rule_ids") or []) if str(x).strip()],
                    match_keywords=[str(x) for x in (row.get("match_keywords") or []) if str(x).strip()],
                )
            )
        self._specs = specs

    def find_applicable(self, domain: str, topic: str, diagnostic: Dict[str, Any]) -> List[SymbolicCheckSpec]:
        rid = str((diagnostic or {}).get("rule") or "").strip()
        if not rid:
            return []
        out = [s for s in self._specs if rid in s.match_rule_ids]
        return out
