from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from symbolic.symbolic_catalog import SymbolicCheckSpec


class SymbolicExperienceBank:
    """Store rule-level symbolic experience without polluting the curated catalog.

    Agentic proposals are first written here. Once the same proposal is observed
    multiple times for the same (domain, topic, rule), it can be reused as a
    promoted bottom-up spec in later runs.
    """

    def __init__(self, path: str = "results/rule_experience_bank.json", promotion_threshold: int = 2) -> None:
        self.path = Path(path)
        self.promotion_threshold = max(1, int(promotion_threshold))

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"rules": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("rules", {})
                return data
        except Exception:
            pass
        return {"rules": {}}

    def save(self, data: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_promoted_specs(self, domain: str, topic: str, rule_id: str) -> List[SymbolicCheckSpec]:
        data = self.load()
        key = self._rule_key(domain, topic, rule_id)
        entry = (data.get("rules") or {}).get(key) or {}
        proposals = entry.get("proposed_specs") or {}

        promoted: List[SymbolicCheckSpec] = []
        for item in proposals.values():
            if not isinstance(item, dict):
                continue
            if int(item.get("count", 0) or 0) < self.promotion_threshold:
                continue
            payload = item.get("spec")
            if not isinstance(payload, dict):
                continue
            try:
                promoted.append(SymbolicCheckSpec(**payload))
            except Exception:
                continue
        return promoted

    def record_event(
        self,
        *,
        domain: str,
        topic: str,
        rule_id: str,
        diagnostic: Dict[str, Any],
        outcome: str,
        had_symbolic_match: bool,
        spec_ids: List[str],
        proposed_specs: List[SymbolicCheckSpec],
    ) -> None:
        data = self.load()
        rules = data.setdefault("rules", {})
        key = self._rule_key(domain, topic, rule_id)
        entry = rules.setdefault(
            key,
            {
                "domain": domain,
                "topic": topic,
                "rule_id": rule_id,
                "events": 0,
                "outcomes": {},
                "no_symbolic_match": 0,
                "seen_spec_ids": {},
                "proposed_specs": {},
                "recent_messages": [],
            },
        )

        entry["events"] = int(entry.get("events", 0) or 0) + 1
        outcomes = entry.setdefault("outcomes", {})
        outcomes[outcome] = int(outcomes.get(outcome, 0) or 0) + 1
        if not had_symbolic_match:
            entry["no_symbolic_match"] = int(entry.get("no_symbolic_match", 0) or 0) + 1

        seen_spec_ids = entry.setdefault("seen_spec_ids", {})
        for spec_id in spec_ids:
            if not spec_id:
                continue
            seen_spec_ids[spec_id] = int(seen_spec_ids.get(spec_id, 0) or 0) + 1

        proposed_bucket = entry.setdefault("proposed_specs", {})
        for spec in proposed_specs:
            payload = {
                "spec_id": spec.spec_id,
                "title": spec.title,
                "description": spec.description,
                "primitive": spec.primitive,
                "params": spec.params,
                "match_rule_ids": list(spec.match_rule_ids or []),
                "match_keywords": list(spec.match_keywords or []),
            }
            item = proposed_bucket.setdefault(spec.spec_id, {"count": 0, "spec": payload})
            item["count"] = int(item.get("count", 0) or 0) + 1
            item["spec"] = payload

        message = str((diagnostic or {}).get("message") or "").strip()
        if message:
            recent_messages = entry.setdefault("recent_messages", [])
            if message not in recent_messages:
                recent_messages.append(message)
            if len(recent_messages) > 10:
                del recent_messages[:-10]

        self.save(data)

    def _rule_key(self, domain: str, topic: str, rule_id: str) -> str:
        return f"{domain}::{topic}::{rule_id}"