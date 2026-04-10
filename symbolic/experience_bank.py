from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, List

from symbolic.symbolic_catalog import SymbolicCheckSpec


class SymbolicExperienceBank:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._events: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                self._events = data
        except Exception:
            self._events = []

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._events, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_promoted_specs(self, domain: str, topic: str, rule_id: str) -> List[SymbolicCheckSpec]:
        # Conservative default: no automatic promotion unless explicitly implemented.
        return []

    def record_event(
        self,
        *,
        sample_id: str = "",
        domain: str,
        topic: str,
        rule_id: str,
        diagnostic: Dict[str, Any] | None = None,
        outcome: str,
        had_symbolic_match: bool = False,
        spec_ids: List[str] | None = None,
        proposed_specs: List[SymbolicCheckSpec],
    ) -> None:
        safe_spec_ids = [str(s) for s in (spec_ids or []) if str(s).strip()]
        payload: Dict[str, Any] = {
            "sample_id": sample_id,
            "domain": domain,
            "topic": topic,
            "rule_id": rule_id,
            "outcome": outcome,
            "had_symbolic_match": bool(had_symbolic_match),
            "spec_ids": safe_spec_ids,
            "proposed_specs": [asdict(s) for s in proposed_specs],
        }
        if isinstance(diagnostic, dict):
            payload["diagnostic"] = {
                "rule": diagnostic.get("rule"),
                "message": diagnostic.get("message"),
                "evidence": diagnostic.get("evidence"),
            }

        self._events.append(
            payload
        )
        # Keep bounded to avoid runaway growth.
        if len(self._events) > 5000:
            self._events = self._events[-5000:]
        self._save()
