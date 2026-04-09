from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def _normalize_topic_key(domain: str, topic: str) -> str:
    d = str(domain or "Unknown").strip().lower()
    t_raw = str(topic or "Unknown").strip()
    t = t_raw.split("/")[-1].strip().lower() if "/" in t_raw else t_raw.lower()
    return f"{d}::{t}"


class ExperienceCodeEngine:
    """Execute generated experience checks mapped by rule_id.

    The engine is deterministic and only uses pre-generated Python functions.
    """

    def __init__(
        self,
        *,
        manifest_path: str = "results/experience_symbolic_program_manifest_300.json",
        module_name: str = "symbolic.generated_experience_checks",
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.module_name = module_name
        self._registry_by_rule_id: Dict[str, Dict[str, Any]] = {}
        self._registry_by_topic: Dict[str, List[Dict[str, Any]]] = {}
        self._functions: Dict[str, Callable[[dict], dict]] = {}
        self._available = False
        self._load()

    @property
    def available(self) -> bool:
        return self._available

    def _load(self) -> None:
        if not self.manifest_path.exists():
            self._available = False
            return

        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception:
            self._available = False
            return

        checks = manifest.get("checks") if isinstance(manifest, dict) else []
        if not isinstance(checks, list):
            self._available = False
            return

        try:
            mod = importlib.import_module(self.module_name)
        except Exception:
            self._available = False
            return

        by_rule: Dict[str, Dict[str, Any]] = {}
        by_topic: Dict[str, List[Dict[str, Any]]] = {}
        funcs: Dict[str, Callable[[dict], dict]] = {}

        for item in checks:
            if not isinstance(item, dict):
                continue
            rule_id = str(item.get("rule_id") or "").strip()
            func_name = str(item.get("function_name") or "").strip()
            if not rule_id or not func_name:
                continue
            fn = getattr(mod, func_name, None)
            if not callable(fn):
                continue

            rec = {
                "rule_id": rule_id,
                "domain": str(item.get("domain") or "Unknown"),
                "topic": str(item.get("topic") or "Unknown"),
                "function_name": func_name,
            }
            by_rule[rule_id] = rec
            topic_key = _normalize_topic_key(rec["domain"], rec["topic"])
            by_topic.setdefault(topic_key, []).append(rec)
            funcs[rule_id] = fn

        self._registry_by_rule_id = by_rule
        self._registry_by_topic = by_topic
        self._functions = funcs
        self._available = bool(self._functions)

    def has_rule(self, rule_id: str) -> bool:
        return str(rule_id or "") in self._functions

    def list_topic_rule_ids(self, domain: str, topic: str) -> List[str]:
        key = _normalize_topic_key(domain, topic)
        return [str(x.get("rule_id")) for x in self._registry_by_topic.get(key, []) if x.get("rule_id")]

    def run_rule(self, rule_id: str, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        rid = str(rule_id or "")
        fn = self._functions.get(rid)
        if fn is None:
            return None

        try:
            out = fn(sample)
        except Exception as exc:
            return {
                "result": "inconclusive",
                "message": f"experience code runtime error: {exc}",
                "evidence": "",
            }

        if not isinstance(out, dict):
            return {
                "result": "inconclusive",
                "message": "experience code returned non-dict",
                "evidence": "",
            }

        result = str(out.get("result") or "inconclusive")
        if result not in {"pass", "fail", "inconclusive"}:
            result = "inconclusive"

        return {
            "result": result,
            "message": str(out.get("message") or ""),
            "evidence": str(out.get("evidence") or ""),
        }

    def run_topic_checks(self, domain: str, topic: str, sample: Dict[str, Any], limit: int = 40) -> List[Tuple[str, Dict[str, Any]]]:
        key = _normalize_topic_key(domain, topic)
        out: List[Tuple[str, Dict[str, Any]]] = []
        for rec in self._registry_by_topic.get(key, [])[: max(0, int(limit))]:
            rid = str(rec.get("rule_id") or "")
            if not rid:
                continue
            result = self.run_rule(rid, sample)
            if result is None:
                continue
            out.append((rid, result))
        return out
