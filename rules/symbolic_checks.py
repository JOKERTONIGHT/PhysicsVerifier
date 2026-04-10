from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GeneratedSymbolicCheckSpec:
    spec_id: str
    title: str
    description: str
    primitive: str
    params: Dict[str, Any] = field(default_factory=dict)
    source_rule_id: Optional[str] = None
    source_message_substring: Optional[str] = None


class GeneratedSymbolicCheckRegistry:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._items: Dict[str, GeneratedSymbolicCheckSpec] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return
            for row in data:
                if not isinstance(row, dict):
                    continue
                sid = str(row.get("spec_id") or "").strip()
                if not sid:
                    continue
                self._items[sid] = GeneratedSymbolicCheckSpec(**row)
        except Exception:
            self._items = {}

    def _save(self) -> None:
        payload = [asdict(v) for v in self._items.values()]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def upsert(self, spec: GeneratedSymbolicCheckSpec) -> None:
        self._items[spec.spec_id] = spec
        self._save()


class GeneratedSymbolicCheckExecutor:
    def run(self, ctx: Any, specs: List[GeneratedSymbolicCheckSpec]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        text_all = str(getattr(ctx, "text_all", "") or "")

        for spec in specs:
            primitive = str(spec.primitive or "").strip().lower()
            result = "inconclusive"

            # Conservative lightweight checks: only mark fail when there is explicit missing token evidence.
            if primitive == "formula_pattern":
                req = [str(x) for x in (spec.params.get("required_symbols") or []) if str(x).strip()]
                if req and not all(r in text_all for r in req):
                    result = "fail"
                elif req:
                    result = "inconclusive"
            elif primitive in {"equation_equivalence", "inequality_consistency", "required_symbols"}:
                result = "inconclusive"

            out.append(
                {
                    "spec_id": spec.spec_id,
                    "primitive": spec.primitive,
                    "title": spec.title,
                    "symbolic_result": result,
                    "rule": f"symbolic::{spec.spec_id}",
                    "symbol": None,
                    "message": spec.description,
                    "evidence": None,
                    "details": {"source_rule_id": spec.source_rule_id},
                }
            )
        return out


def catalog_spec_to_generated(spec: Any) -> GeneratedSymbolicCheckSpec:
    return GeneratedSymbolicCheckSpec(
        spec_id=str(getattr(spec, "spec_id", "")),
        title=str(getattr(spec, "title", "")),
        description=str(getattr(spec, "description", "")),
        primitive=str(getattr(spec, "primitive", "")),
        params=dict(getattr(spec, "params", {}) or {}),
        source_rule_id=None,
        source_message_substring=None,
    )
