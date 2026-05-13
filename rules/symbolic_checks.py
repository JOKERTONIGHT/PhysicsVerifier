from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
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


_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_LATEX_PUNCT_RE = re.compile(r"[\\{}$^_~`]")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text_for_match(text: str) -> str:
    """Lower-case + strip LaTeX command tokens / braces so tokens can be loosely compared."""
    if not text:
        return ""
    t = text.lower()
    t = _LATEX_CMD_RE.sub(" ", t)
    t = _LATEX_PUNCT_RE.sub(" ", t)
    t = _WHITESPACE_RE.sub(" ", t)
    return t.strip()


_SHORT_TOKEN_BOUNDARY_RE_CACHE: Dict[str, re.Pattern] = {}


def _token_present(token: str, text_lower: str, text_norm: str) -> bool:
    """Detect whether ``token`` appears in ``text``.

    Short or single-letter tokens (e.g. "M", "r", "x") would otherwise spuriously
    match inside ordinary words ("text" contains "x"), which lets the symbolic
    executor mistakenly claim coverage. For tokens shorter than 4 alphanumeric
    characters we require a word/non-letter boundary on both sides.
    """
    if not token:
        return False
    tl = str(token).lower()
    if not tl:
        return False
    tn = _normalize_text_for_match(token)
    requires_boundary = len(re.sub(r"[^A-Za-z0-9]", "", tn)) <= 3
    if requires_boundary and tn:
        pat = _SHORT_TOKEN_BOUNDARY_RE_CACHE.get(tn)
        if pat is None:
            pat = re.compile(r"(?<![A-Za-z0-9])" + re.escape(tn) + r"(?![A-Za-z0-9])")
            _SHORT_TOKEN_BOUNDARY_RE_CACHE[tn] = pat
        return bool(pat.search(text_norm))
    if tl in text_lower:
        return True
    return bool(tn) and tn in text_norm


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
    """Lightweight executor for symbolic checks.

    Result semantics (consumed by ``PhysicsRuleVerifier`` reconciliation):
      * ``pass``         -> the canonical *correct* pattern is present in the
                            student's response, so the original LLM diagnostic
                            does **not** apply -> SUPPRESS the diagnostic.
      * ``fail``         -> the canonical pattern is meaningfully absent /
                            contradicted, supporting the diagnostic.
      * ``inconclusive`` -> evidence is mixed or missing; keep the diagnostic
                            but mark it as not strongly substantiated.

    Previously the executor never returned ``pass`` (precluding any
    suppression) and treated "no required tokens at all" as ``fail``. Both of
    those behaviours hurt precision: the off-topic rules that retrieved poorly
    were still being labelled "supported" by symbolic checks even though the
    canonical formula never appeared in the response. The new behaviour is
    more discriminating: full token coverage / canonical match -> ``pass``;
    partial / no coverage -> ``inconclusive``.
    """

    def run(self, ctx: Any, specs: List[GeneratedSymbolicCheckSpec]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        text_all = str(getattr(ctx, "text_all", "") or "")
        text_lower = text_all.lower()
        text_norm = _normalize_text_for_match(text_all)

        for spec in specs:
            primitive = str(spec.primitive or "").strip().lower()
            params = spec.params or {}
            result, details = self._evaluate(primitive, params, text_lower, text_norm)
            details = dict(details or {})
            details.setdefault("source_rule_id", spec.source_rule_id)

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
                    "details": details,
                }
            )
        return out

    def _evaluate(
        self,
        primitive: str,
        params: Dict[str, Any],
        text_lower: str,
        text_norm: str,
    ) -> tuple:
        details: Dict[str, Any] = {}

        if primitive == "formula_pattern":
            req = [str(x) for x in (params.get("required_symbols") or []) if str(x).strip()]
            if not req:
                return "inconclusive", details
            present = [t for t in req if _token_present(t, text_lower, text_norm)]
            missing = [t for t in req if t not in present]
            details["matched_tokens"] = present
            details["missing_tokens"] = missing
            if not missing:
                return "pass", details
            if not present:
                return "inconclusive", details
            return "inconclusive", details

        if primitive in {"equation_equivalence", "inequality_consistency"}:
            req = [str(x) for x in (params.get("required_symbols") or []) if str(x).strip()]
            canonicals = params.get("canonical_latex") or []
            if isinstance(canonicals, str):
                canonicals = [canonicals]
            for cand in canonicals:
                cn = _normalize_text_for_match(str(cand))
                if cn and len(cn) >= 4 and cn in text_norm:
                    details["canonical_match"] = str(cand)
                    return "pass", details
            if req:
                present = [t for t in req if _token_present(t, text_lower, text_norm)]
                details["matched_tokens"] = present
                if len(present) >= max(2, int(len(req) * 0.66)):
                    # All-or-nearly-all symbols present and canonical missing: structural mismatch
                    details["coverage"] = round(len(present) / max(1, len(req)), 3)
                    return "fail", details
            return "inconclusive", details

        if primitive == "power_law":
            dep = [str(x) for x in (params.get("dependent_candidates") or []) if str(x).strip()]
            ind = [str(x) for x in (params.get("independent_candidates") or []) if str(x).strip()]
            dep_hit = any(_token_present(d, text_lower, text_norm) for d in dep)
            ind_hit = any(_token_present(i, text_lower, text_norm) for i in ind)
            details["dep_hit"] = dep_hit
            details["ind_hit"] = ind_hit
            if dep_hit and ind_hit:
                return "inconclusive", details
            if not dep_hit and not ind_hit:
                return "inconclusive", details
            return "fail", details

        if primitive == "compliance_phrase":
            phrases = [str(x) for x in (params.get("phrases") or []) if str(x).strip()]
            if not phrases:
                return "inconclusive", details
            matched = [p for p in phrases if p.lower() in text_lower]
            details["matched_phrases"] = matched
            if matched:
                return "pass", details
            return "inconclusive", details

        if primitive == "required_symbols":
            req = [str(x) for x in (params.get("required_symbols") or []) if str(x).strip()]
            if not req:
                return "inconclusive", details
            present = [t for t in req if _token_present(t, text_lower, text_norm)]
            details["matched_tokens"] = present
            if len(present) == len(req):
                return "pass", details
            if not present:
                return "fail", details
            return "inconclusive", details

        return "inconclusive", details


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
