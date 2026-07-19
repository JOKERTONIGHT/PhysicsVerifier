"""Catalog-building retrieval helpers (from unified rules pipeline PR).

Used by offline catalog scripts only; runtime verifier uses core.rule_catalog_retrieval.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple

from core.rule_catalog_retrieval import (
    TOKEN_RE,
    extract_keywords,
    keep_token,
    norm_text,
    ordered_unique,
    GENERIC_SYMBOLS,
    META_RULE_HINTS,
    GENERIC_RULE_SIGNAL_TERMS,
    GENERIC_SCENE_PARTS,
)

GENERIC_SCENE_TERMS = {
    "atomic",
    "application",
    "applications",
    "body",
    "bodies",
    "charge",
    "charges",
    "collision",
    "collisions",
    "current",
    "currents",
    "dynamics",
    "electric",
    "electromagnetism",
    "energy",
    "equation",
    "equations",
    "field",
    "fields",
    "force",
    "forces",
    "function",
    "functions",
    "induction",
    "interaction",
    "interactions",
    "law",
    "laws",
    "light",
    "magnetic",
    "mass",
    "mechanics",
    "motion",
    "optics",
    "particle",
    "particles",
    "physics",
    "potential",
    "radiation",
    "role",
    "statistical",
    "system",
    "systems",
    "temperature",
    "thermodynamics",
    "time",
    "transfer",
    "wave",
    "waves",
    "电学",
    "力学",
    "场",
    "定律",
    "方程",
    "热学",
    "物理",
    "电磁",
    "粒子",
    "能量",
    "运动",
}

LOW_SIGNAL_KEYWORDS = {
    "about",
    "all",
    "along",
    "area",
    "axis",
    "being",
    "body",
    "center",
    "com",
    "constant",
    "const",
    "current",
    "distance",
    "dynamics",
    "energy",
    "equal",
    "field",
    "fields",
    "function",
    "initial",
    "first",
    "length",
    "position",
    "mass",
    "model",
    "models",
    "motion",
    "number",
    "numbers",
    "object",
    "objects",
    "per",
    "physical",
    "rate",
    "results",
    "surface",
    "sum",
    "system",
    "systems",
    "term",
    "terms",
    "theory",
    "time",
    "total",
    "use",
    "used",
    "volume",
    "whether",
    "very",
    "short",
    "opening",
    "vec",
    "all",
    "no",
    "relations",
    "relation",
    "applications",
    "principles",
    "second",
}

GENERIC_MATH_RULE_HINTS = {
    "cos",
    "cot",
    "csc",
    "identity",
    "root",
    "sec",
    "sin",
    "sqrt",
    "substitution",
    "tan",
    "三角",
    "代换",
    "恒等式",
    "根式",
}

PHYSICAL_CONTEXT_HINTS = {
    "ammeter",
    "atom",
    "battery",
    "beam",
    "black hole",
    "camera",
    "capacitor",
    "charge",
    "circuit",
    "coil",
    "collision",
    "current",
    "dipole",
    "disc",
    "earth",
    "electric",
    "electron",
    "field",
    "flux",
    "frog",
    "gas",
    "gravity",
    "induction",
    "interferometer",
    "laser",
    "lens",
    "loop",
    "magnetic",
    "mass",
    "orbit",
    "particle",
    "pinhole",
    "planet",
    "pressure",
    "relativity",
    "resistor",
    "rod",
    "satellite",
    "spring",
    "string",
    "superconduct",
    "torque",
    "voltage",
    "wave",
}

GENERIC_SCENE_PARTS = {
    "applications",
    "first",
    "law",
    "laws",
    "models",
    "no",
    "not",
    "principles",
    "relation",
    "relations",
    "second",
    "simple",
    "theory",
    "use",
}

GENERIC_ACTION_TERMS = {
    "apply",
    "assume",
    "calculate",
    "check",
    "compute",
    "confirm",
    "derive",
    "determine",
    "ensure",
    "given",
    "judge",
    "open",
    "opening",
    "show",
    "solve",
    "verify",
    "出现",
    "判断",
    "处理",
    "要求",
    "计算",
    "证明",
    "给出",
    "设",
    "说明",
    "需要",
}

NARROW_RULE_HINT_GROUPS = (
    ("binary system", "双星"),
    ("gravitational scattering", "引力散射", "impact parameter", "偏转角", "掠过"),
    ("collision orbit", "碰撞后", "变轨", "近地点", "r_p"),
    ("cherenkov", "切伦科夫"),
    ("mirage", "蜃景", "虚幻湖面", "maupertuis", "fermat"),
)

STRONG_SYMBOL_ALLOWLIST = {
    "R_E",
    "r_p",
    "dE",
    "dr",
    "dx",
    "dt",
    "EMF",
    "ISS",
}

SHORT_TOKEN_ALLOWLIST = {
    "ISS",
    "EMF",
}

def is_low_signal_term(value: str) -> bool:
    return norm_text(value).casefold() in LOW_SIGNAL_KEYWORDS

def is_generic_scene_term(value: str) -> bool:
    return norm_text(value).casefold() in GENERIC_SCENE_TERMS

def is_generic_scene_part(value: str) -> bool:
    return norm_text(value).casefold() in GENERIC_SCENE_PARTS

def is_generic_rule_signal(value: str) -> bool:
    normalized = norm_text(value).casefold()
    if not normalized:
        return False
    if normalized in GENERIC_RULE_SIGNAL_TERMS:
        return True
    return any(term in normalized for term in GENERIC_RULE_SIGNAL_TERMS)

def is_short_token_allowed(value: str) -> bool:
    return norm_text(value).upper() in SHORT_TOKEN_ALLOWLIST

def _remove_keywords(values: Iterable[str], blocked: Iterable[str]) -> List[str]:
    blocked_keys = {norm_text(item).casefold() for item in blocked if norm_text(item)}
    return [
        item
        for item in ordered_unique(values)
        if norm_text(item).casefold() not in blocked_keys
    ]

def remove_generic_rule_keywords(values: Iterable[str]) -> List[str]:
    return _remove_keywords(values, GENERIC_RULE_SIGNAL_TERMS)

def _rule_text(*, title: str, trigger: str, check_logic: str) -> str:
    return " ".join([norm_text(title), norm_text(trigger), norm_text(check_logic)]).strip()

def _count_hint_hits(text: str, hints: Iterable[str]) -> int:
    normalized = norm_text(text).casefold()
    if not normalized:
        return 0
    return sum(
        1
        for hint in hints
        if (candidate := norm_text(hint).casefold()) and candidate in normalized
    )

def is_generic_math_rule(*, title: str, trigger: str, check_logic: str) -> bool:
    text = _rule_text(title=title, trigger=trigger, check_logic=check_logic)
    if not text:
        return False
    math_hits = _count_hint_hits(text, GENERIC_MATH_RULE_HINTS)
    physical_hits = _count_hint_hits(text, PHYSICAL_CONTEXT_HINTS)
    return math_hits >= 2 and physical_hits == 0

def is_narrow_applicability_rule(*, title: str, trigger: str, check_logic: str) -> bool:
    text = _rule_text(title=title, trigger=trigger, check_logic=check_logic).casefold()
    if not text:
        return False
    for hints in NARROW_RULE_HINT_GROUPS:
        if all(hint.casefold() in text for hint in hints):
            return True
    # Accept partial matches for longer narrow phrases.
    return any(
        sum(1 for hint in hints if hint.casefold() in text) >= 2
        for hints in NARROW_RULE_HINT_GROUPS
        if len(hints) >= 2
    )

def is_strong_symbol(value: str) -> bool:
    item = norm_text(value)
    if not item:
        return False
    if item in STRONG_SYMBOL_ALLOWLIST:
        return True
    if len(item) == 1 and item in GENERIC_SYMBOLS:
        return False
    if re.fullmatch(r"[A-Za-z]{1,2}", item) and not is_short_token_allowed(item):
        return False
    if any(ch in item for ch in ("_", "/", "\\", "(", ")", "^")):
        return True
    if re.search(r"\d", item):
        return True
    if len(item) >= 4:
        return True
    return item.isupper() and len(item) >= 3

def classify_rule_retrieval_profile(*, title: str, trigger: str, check_logic: str) -> Dict[str, Any]:
    text = _rule_text(title=title, trigger=trigger, check_logic=check_logic)
    lowered = text.casefold()
    generic_math_rule = is_generic_math_rule(title=title, trigger=trigger, check_logic=check_logic)
    narrow_rule = is_narrow_applicability_rule(title=title, trigger=trigger, check_logic=check_logic)
    meta_hint_rule = bool(lowered) and any(hint.casefold() in lowered for hint in META_RULE_HINTS)
    return {
        "text": text,
        "generic_math_rule": generic_math_rule,
        "narrow_rule": narrow_rule,
        "meta_hint_rule": meta_hint_rule,
    }

def normalize_rule_for_retrieval(rule: Dict[str, Any]) -> Dict[str, Any]:
    patched = dict(rule)
    profile = classify_rule_retrieval_profile(
        title=str(rule.get("title") or ""),
        trigger=str(rule.get("trigger") or ""),
        check_logic=str(rule.get("check_logic") or ""),
    )
    generic_math_rule = bool(profile["generic_math_rule"])
    narrow_rule = bool(profile["narrow_rule"])

    retrieval_flags = dict(patched.get("retrieval_flags") or {})
    retrieval_flags["generic_math_rule"] = bool(generic_math_rule)
    retrieval_flags["narrow_rule"] = bool(narrow_rule)
    patched["retrieval_flags"] = retrieval_flags

    match_features = dict(patched.get("match_features") or {})
    if match_features:
        match_features["trigger_keywords"] = remove_generic_rule_keywords(match_features.get("trigger_keywords") or [])
        match_features["object_keywords"] = remove_generic_rule_keywords(match_features.get("object_keywords") or [])
        patched["match_features"] = match_features
    if generic_math_rule or narrow_rule:
        patched["scope"] = "meta"
    return patched

def _is_physical_context_token(token: str) -> bool:
    normalized = norm_text(token).casefold()
    if not normalized:
        return False
    if normalized in PHYSICAL_CONTEXT_HINTS:
        return True
    return any(normalized in hint or hint in normalized for hint in PHYSICAL_CONTEXT_HINTS if " " not in hint)

def refine_topic_hints(
    *,
    scene_keywords: Iterable[str],
    topic_keywords: Iterable[str],
    rule_texts: Iterable[str],
) -> Tuple[List[str], List[str]]:
    derived_rule_keywords = extract_keywords(rule_texts, max_keywords=16)
    keywords = ordered_unique(list(topic_keywords) + list(derived_rule_keywords))
    keywords = [
        item
        for item in keywords
        if not is_low_signal_term(item)
        and not is_generic_scene_term(item)
        and not (norm_text(item).isalpha() and len(norm_text(item)) <= 2)
    ]
    scene = [
        item
        for item in ordered_unique(scene_keywords)
        if not is_low_signal_term(item)
        and not (norm_text(item).casefold() in {"resistance", "length", "time", "mass", "current"})
    ]
    filtered_scene: List[str] = []
    for item in scene:
        tokens = [tok for tok in TOKEN_RE.findall(item) if keep_token(tok)]
        if not tokens:
            continue
        context_count = sum(1 for tok in tokens if _is_physical_context_token(tok))
        lowered = item.casefold()
        action_heavy = any(term in lowered for term in GENERIC_ACTION_TERMS)
        has_cjk = bool(re.search(r"[\u4e00-\u9fff]", item))
        if lowered.endswith("时") and context_count < 2:
            continue
        if " or " in lowered or " 或 " in item:
            continue
        if len(tokens) >= 5 and context_count < 2:
            continue
        if len(norm_text(item)) >= 20 and context_count == 0:
            continue
        if action_heavy and context_count < 2 and (" " in item or len(norm_text(item)) >= 10):
            continue
        if action_heavy and context_count < 3 and len(tokens) >= 3:
            continue
        if has_cjk and context_count < 2 and len(norm_text(item)) >= 8:
            continue
        filtered_scene.append(item)
    scene = filtered_scene
    scene = sorted(
        scene,
        key=lambda item: (
            -min(sum(1 for tok in TOKEN_RE.findall(item) if _is_physical_context_token(tok)), 2),
            -(1 if 2 <= len([tok for tok in TOKEN_RE.findall(item) if keep_token(tok)]) <= 3 else 0),
            len(norm_text(item)),
            item.casefold(),
        ),
    )
    return scene[:16], keywords[:20]
