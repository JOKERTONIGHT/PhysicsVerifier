from __future__ import annotations

import re
from typing import Any, Dict, List, Set


_SYMBOL_ALIASES = {
    "tau": {"τ", "tau"},
    "omega": {"ω", "omega"},
    "theta": {"θ", "theta"},
    "phi": {"φ", "phi"},
    "pi": {"π", "pi"},
    "delta": {"δ", "delta"},
    "gamma": {"γ", "gamma"},
    "lambda": {"λ", "lambda"},
    "rms": {"rms", "effective", "有效值"},
    "avg": {"avg", "average", "mean", "平均值"},
}


def normalize_math_text(text: Any) -> str:
    out = str(text or "")
    for token in ["\\left", "\\right", "{", "}", "$", " "]:
        out = out.replace(token, "")
    return out.lower()


def normalize_symbol(symbol: Any) -> str:
    return normalize_math_text(symbol)


def symbol_aliases(symbol: Any) -> Set[str]:
    base = normalize_symbol(symbol)
    out: Set[str] = {base} if base else set()
    if not base:
        return out

    if base in _SYMBOL_ALIASES:
        out.update({_normalize_alias(v) for v in _SYMBOL_ALIASES[base]})

    for k, vals in _SYMBOL_ALIASES.items():
        norm_vals = {_normalize_alias(v) for v in vals}
        if base in norm_vals:
            out.add(k)
            out.update(norm_vals)
            break

    return {v for v in out if v}


def _normalize_alias(value: Any) -> str:
    return normalize_math_text(value)


def text_has_symbol(text: Any, symbol: Any) -> bool:
    hay = normalize_math_text(text)
    if not hay:
        return False

    for candidate in symbol_aliases(symbol):
        if not candidate:
            continue
        if len(candidate) == 1 and candidate.isalpha():
            pat = re.compile(rf"(^|[^a-zA-Z0-9_]){re.escape(candidate)}([^a-zA-Z0-9_]|$)", re.I)
            if pat.search(str(text or "")):
                return True
            continue
        if candidate in hay:
            return True
    return False


def symbol_match_report(text: Any, required_symbols: List[str], min_ratio: float = 0.5) -> Dict[str, Any]:
    required = [str(s) for s in (required_symbols or []) if str(s).strip()]
    if not required:
        return {
            "required": [],
            "matched": [],
            "missing": [],
            "ratio": 1.0,
            "ok": True,
        }

    matched: List[str] = []
    missing: List[str] = []
    for sym in required:
        if text_has_symbol(text, sym):
            matched.append(sym)
        else:
            missing.append(sym)

    ratio = len(matched) / max(1, len(required))
    return {
        "required": required,
        "matched": matched,
        "missing": missing,
        "ratio": ratio,
        "ok": ratio >= float(min_ratio),
    }
