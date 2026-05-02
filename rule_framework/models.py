from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class RulePath:
    domain: str
    topic: str
    context: str = "general"
    cluster: str = "unclustered"

    def as_dict(self) -> Dict[str, str]:
        return {
            "domain": self.domain,
            "topic": self.topic,
            "context": self.context,
            "cluster": self.cluster,
        }


@dataclass
class BuildConfig:
    cluster_topic_threshold: int = 12
    cluster_bucket_threshold: int = 3
    include_flat_rules: bool = True
    include_rule_tree: bool = True


@dataclass
class MaintenanceResult:
    catalog: Dict[str, Any]
    changed_rule_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings}
