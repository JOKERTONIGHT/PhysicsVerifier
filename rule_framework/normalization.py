from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Iterable, List

from core.rule_catalog_retrieval import extract_keywords, norm_text, ordered_unique


def norm_key(value: Any) -> str:
    return norm_text(value).casefold()


def normalize_topic(domain: str, topic: str) -> str:
    norm_domain = norm_text(domain)
    norm_topic = norm_text(topic)
    if "/" in norm_topic:
        left, right = [part.strip() for part in norm_topic.split("/", 1)]
        if left.casefold() == norm_domain.casefold():
            return right
    return norm_topic


def topic_key(domain: str, topic: str) -> str:
    return f"{norm_key(domain)}::{norm_key(topic)}"


def safe_symbolic_hint(raw_hint: Any) -> Dict[str, Any]:
    hint = raw_hint if isinstance(raw_hint, dict) else {}
    primitive = norm_text(hint.get("primitive") or "none") or "none"
    canonical = norm_text(hint.get("canonical") or "")
    required_symbols = ordered_unique(str(item) for item in (hint.get("required_symbols") or []))
    return {
        "primitive": primitive,
        "canonical": canonical,
        "required_symbols": required_symbols,
    }


def normalize_match_surface(text: str) -> str:
    """Lightweight surface normalisation for retrieval (not full LaTeX parsing)."""
    raw = norm_text(text or "")
    if not raw:
        return ""
    out = raw
    for pat, repl in (
        (r"\\mathrm\{([^}]*)\}", r"\1"),
        (r"\\text\{([^}]*)\}", r"\1"),
        (r"\\,|\~", " "),
        (r"\s+", " "),
    ):
        out = re.sub(pat, repl, out, flags=re.I)
    return out.strip()


def build_match_features(title: str, trigger: str, check_logic: str, symbolic_hint: Dict[str, Any]) -> Dict[str, Any]:
    required_symbols = ordered_unique(str(item) for item in symbolic_hint.get("required_symbols", []))
    primitive = norm_text(symbolic_hint.get("primitive") or "none") or "none"
    combined = " ".join([norm_text(title), norm_text(trigger), norm_text(check_logic)])
    return {
        "trigger_keywords": extract_keywords([title, trigger], max_keywords=8),
        "object_keywords": extract_keywords([check_logic], max_keywords=8),
        "required_symbols": required_symbols,
        "primitive": primitive,
        "match_text_normalized": normalize_match_surface(combined),
        "negative_keywords": [],
    }


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", norm_text(value).lower()).strip("_")
    return slug or "unknown"


def experience_rule_fingerprint(rule: Dict[str, Any], domain: str, topic: str) -> str:
    parts = [
        norm_text(domain),
        norm_text(topic),
        norm_text(rule.get("title") or ""),
        norm_text(rule.get("trigger") or ""),
        norm_text(rule.get("check_logic") or ""),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def ordered_strings(values: Iterable[Any]) -> List[str]:
    return ordered_unique(str(item) for item in values)
