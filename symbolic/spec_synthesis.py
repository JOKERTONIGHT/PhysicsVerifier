from __future__ import annotations

from typing import Dict, List

from symbolic.symbolic_catalog import SymbolicCheckSpec


class RuleSymbolicSpecSynthesizer:
    def synthesize_topic(self, domain: str, topic: Dict) -> Dict[str, List[SymbolicCheckSpec]]:
        # Conservative placeholder: keep pipeline stable even when no synthesized specs are generated.
        _ = domain
        _ = topic
        return {}
