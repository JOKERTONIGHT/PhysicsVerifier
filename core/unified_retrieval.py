from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "can",
    "check",
    "checks",
    "correct",
    "derive",
    "description",
    "determine",
    "domain",
    "equation",
    "error",
    "errors",
    "for",
    "form",
    "from",
    "given",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "logic",
    "not",
    "must",
    "of",
    "on",
    "or",
    "physics",
    "problem",
    "process",
    "result",
    "rule",
    "rules",
    "should",
    "solution",
    "such",
    "that",
    "the",
    "their",
    "them",
    "then",
    "this",
    "those",
    "through",
    "to",
    "topic",
    "under",
    "along",
    "use",
    "using",
    "verify",
    "when",
    "where",
    "which",
    "with",
    "without",
    "题目",
    "公式",
    "关系",
    "分析",
    "出现",
    "判断",
    "前提",
    "利用",
    "可能",
    "必须",
    "条件",
    "检查",
    "根据",
    "正确",
    "注意",
    "涉及",
    "过程",
    "逻辑",
    "结果",
    "要求",
    "计算",
    "证明",
    "验证",
    "解答",
    "说明",
    "需要",
}

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

META_RULE_HINTS = (
    "chart",
    "data extraction",
    "diagram",
    "dimensional",
    "distinguish given",
    "figure",
    "graph",
    "known/unknown",
    "notation",
    "read graph",
    "semantic",
    "table",
    "unit consistency",
    "区分给定",
    "区分已知",
    "区分给定演化规律",
    "区分待求",
    "图区",
    "图像",
    "图表",
    "图表数据",
    "已知",
    "待求",
    "数据提取",
    "文字描述",
    "标注",
    "量纲",
    "量纲一致",
    "表格",
    "读图",
    "语义",
)

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

GENERIC_RULE_SIGNAL_TERMS = {
    "sin",
    "cos",
    "tan",
    "cot",
    "sec",
    "csc",
    "sqrt",
    "根式",
    "代换",
    "恒等式",
    "三角",
    "三角代换",
    "根式方程",
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

GENERIC_SYMBOLS = {
    "a",
    "A",
    "b",
    "B",
    "c",
    "C",
    "d",
    "D",
    "e",
    "E",
    "f",
    "F",
    "g",
    "G",
    "h",
    "H",
    "i",
    "I",
    "k",
    "K",
    "l",
    "L",
    "m",
    "M",
    "n",
    "N",
    "p",
    "P",
    "q",
    "Q",
    "r",
    "R",
    "s",
    "S",
    "t",
    "T",
    "u",
    "U",
    "v",
    "V",
    "w",
    "W",
    "x",
    "X",
    "y",
    "Y",
    "z",
    "Z",
}


def norm_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def ordered_unique(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        item = norm_text(value)
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def match_phrase_or_symbol(needle: str, haystack: str) -> bool:
    target = norm_text(needle)
    text = norm_text(haystack)
    if not target or not text:
        return False

    if len(target) == 1 and re.fullmatch(r"[A-Za-z]", target):
        pat = re.compile(rf"(^|[^A-Za-z0-9_]){re.escape(target)}([^A-Za-z0-9_]|$)", re.I)
        return bool(pat.search(text))

    return target.casefold() in text.casefold()


def keep_token(token: str) -> bool:
    if not token or token.isdigit():
        return False
    if len(token) == 1 and re.fullmatch(r"[A-Za-z0-9_]", token):
        return False
    if token.casefold() in STOPWORDS:
        return False
    return True


def extract_keywords(texts: Iterable[str], *, max_keywords: int) -> List[str]:
    counter: Counter[str] = Counter()
    first_seen: Dict[str, str] = {}
    order: Dict[str, int] = {}

    for text in texts:
        for raw in TOKEN_RE.findall(norm_text(text)):
            token = raw.strip()
            if not keep_token(token):
                continue
            key = token.casefold()
            counter[key] += 1
            first_seen.setdefault(key, token)
            order.setdefault(key, len(order))

    ranked = sorted(counter.items(), key=lambda item: (-item[1], order[item[0]], item[0]))
    return [first_seen[key] for key, _ in ranked[:max_keywords]]


def _remove_keywords(values: Iterable[str], blocked: Iterable[str]) -> List[str]:
    blocked_keys = {norm_text(item).casefold() for item in blocked if norm_text(item)}
    return [item for item in ordered_unique(values) if norm_text(item).casefold() not in blocked_keys]


def _count_hint_hits(text: str, hints: Iterable[str]) -> int:
    lowered = norm_text(text).casefold()
    return sum(1 for hint in hints if hint in lowered)


def is_generic_math_rule(*, title: str, trigger: str, check_logic: str) -> bool:
    text = " ".join([norm_text(title), norm_text(trigger), norm_text(check_logic)])
    if not text:
        return False
    math_hits = _count_hint_hits(text, GENERIC_MATH_RULE_HINTS)
    physical_hits = _count_hint_hits(text, PHYSICAL_CONTEXT_HINTS)
    return math_hits >= 2 and physical_hits == 0


def normalize_rule_for_retrieval(rule: Dict[str, Any]) -> Dict[str, Any]:
    patched = dict(rule)
    if not is_generic_math_rule(
        title=str(rule.get("title") or ""),
        trigger=str(rule.get("trigger") or ""),
        check_logic=str(rule.get("check_logic") or ""),
    ):
        return patched

    patched["scope"] = "meta"
    match_features = dict(patched.get("match_features") or {})
    if match_features:
        trigger_keywords = _remove_keywords(match_features.get("trigger_keywords") or [], GENERIC_RULE_SIGNAL_TERMS)
        object_keywords = _remove_keywords(match_features.get("object_keywords") or [], GENERIC_RULE_SIGNAL_TERMS)
        normalized_text = " ".join(
            [
                norm_text(rule.get("title") or ""),
                norm_text(rule.get("trigger") or ""),
                norm_text(rule.get("check_logic") or ""),
            ]
        ).casefold()
        prepend_trigger: List[str] = []
        prepend_object: List[str] = []
        if "substitution" in normalized_text or "代换" in normalized_text:
            prepend_trigger.append("trig substitution")
        if "root" in normalized_text or "根式" in normalized_text:
            prepend_object.append("root equation")
        match_features["trigger_keywords"] = ordered_unique(prepend_trigger + trigger_keywords)
        match_features["object_keywords"] = ordered_unique(prepend_object + object_keywords)
        patched["match_features"] = match_features
    return patched


def classify_rule_scope(*, title: str, trigger: str, check_logic: str) -> str:
    if is_generic_math_rule(title=title, trigger=trigger, check_logic=check_logic):
        return "meta"
    text = " ".join([norm_text(title), norm_text(trigger), norm_text(check_logic)]).casefold()
    if not text:
        return "domain"
    for hint in META_RULE_HINTS:
        if hint.casefold() in text:
            return "meta"
    return "domain"


def _is_physical_context_token(token: str) -> bool:
    normalized = norm_text(token).casefold()
    if not normalized:
        return False
    if normalized in PHYSICAL_CONTEXT_HINTS:
        return True
    return any(normalized in hint or hint in normalized for hint in PHYSICAL_CONTEXT_HINTS if " " not in hint)


def _scene_anchor_candidates(text: str) -> List[str]:
    tokens = [
        tok
        for tok in TOKEN_RE.findall(norm_text(text))
        if keep_token(tok)
        and (tok.casefold() not in LOW_SIGNAL_KEYWORDS or _is_physical_context_token(tok))
        and tok.casefold() not in GENERIC_SCENE_PARTS
    ]
    if not tokens:
        return []

    anchors: List[str] = []
    for token in tokens:
        lowered = token.casefold()
        if _is_physical_context_token(token) and lowered not in GENERIC_SCENE_TERMS and len(token) >= 5:
            anchors.append(token)

    for size in range(2, min(4, len(tokens)) + 1):
        for start in range(0, len(tokens) - size + 1):
            window = tokens[start : start + size]
            lowered = [tok.casefold() for tok in window]
            if any(item in GENERIC_MATH_RULE_HINTS for item in lowered):
                continue
            context_count = sum(1 for tok in window if _is_physical_context_token(tok))
            if context_count <= 0:
                continue
            content_tokens = [
                item
                for item in lowered
                if item not in GENERIC_SCENE_TERMS and item not in GENERIC_SCENE_PARTS and item not in LOW_SIGNAL_KEYWORDS
            ]
            if len(content_tokens) <= 0:
                continue
            if size >= 3 and len(content_tokens) < 2 and context_count < 2:
                continue
            anchors.append(" ".join(window))

    return ordered_unique(anchors)


def build_scene_keywords(
    *,
    topic_name: str,
    tagged_titles: Iterable[str],
    tagged_aliases: Iterable[str],
    rule_texts: Iterable[str] = (),
) -> List[str]:
    phrase_counter: Counter[str] = Counter()
    phrase_display: Dict[str, str] = {}
    phrase_source_bonus: Dict[str, float] = {}
    phrase_order: Dict[str, int] = {}

    def keep_scene_part(text: str, *, allow_single: bool = False) -> bool:
        item = norm_text(text)
        if not item:
            return False
        lowered = item.casefold()
        if lowered in GENERIC_SCENE_TERMS or lowered in GENERIC_SCENE_PARTS or lowered in LOW_SIGNAL_KEYWORDS:
            return False
        if item.isalpha() and item.upper() == item and len(item) <= 2:
            return False
        if not allow_single and " " not in item and len(item) < 4:
            return False
        return True

    tagged_texts = [norm_text(item) for item in tagged_titles]
    alias_texts = [norm_text(item) for item in tagged_aliases]
    rule_phrase_texts = [norm_text(item) for item in rule_texts]
    raw_inputs: List[Tuple[str, float]] = (
        [(topic_name, 2.0)]
        + [(item, 1.25) for item in tagged_texts]
        + [(item, 1.0) for item in alias_texts]
        + [(item, 2.5) for item in rule_phrase_texts]
    )

    def add_phrase(text: str, source_bonus: float, *, allow_single: bool) -> None:
        if not keep_scene_part(text, allow_single=allow_single):
            return
        key = norm_text(text).casefold()
        phrase_counter[key] += 1
        phrase_display.setdefault(key, norm_text(text))
        phrase_source_bonus[key] = max(phrase_source_bonus.get(key, 0.0), source_bonus)
        phrase_order.setdefault(key, len(phrase_order))

    for raw_text, source_bonus in raw_inputs:
        text = norm_text(raw_text)
        if not text:
            continue
        allow_single = raw_text == topic_name
        add_phrase(text, source_bonus, allow_single=allow_single)
        for part in re.split(r"[()/,;:，；]+| and | of | with | - | or | 或 ", text, flags=re.I):
            add_phrase(part, source_bonus - 0.25, allow_single=False)
        for anchor in _scene_anchor_candidates(text):
            add_phrase(anchor, source_bonus + 0.25, allow_single=False)

    def phrase_rank(key: str) -> Tuple[float, int]:
        phrase = phrase_display[key]
        tokens = [tok for tok in TOKEN_RE.findall(phrase) if keep_token(tok)]
        token_count = len(tokens)
        context_bonus = min(sum(1 for tok in tokens if _is_physical_context_token(tok)), 2) * 0.9
        compact_bonus = 1.5 if 2 <= token_count <= 3 else (0.75 if token_count in {1, 4} else 0.0)
        token_bonus = min(token_count, 4) * 0.35
        length_penalty = 1.0 if len(phrase) >= 36 else (0.5 if len(phrase) >= 24 else 0.0)
        score = (
            phrase_source_bonus.get(key, 0.0)
            + context_bonus
            + compact_bonus
            + token_bonus
            + phrase_counter[key]
            - length_penalty
        )
        return (-score, phrase_order[key])

    ranked_keys = sorted(phrase_display.keys(), key=phrase_rank)
    return [phrase_display[key] for key in ranked_keys[:16]]


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
        if norm_text(item).casefold() not in LOW_SIGNAL_KEYWORDS
        and norm_text(item).casefold() not in GENERIC_SCENE_TERMS
        and not (norm_text(item).isalpha() and len(norm_text(item)) <= 2)
    ]
    scene = [
        item
        for item in ordered_unique(scene_keywords)
        if norm_text(item).casefold() not in LOW_SIGNAL_KEYWORDS
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


def build_topic_required_symbols(rules: Iterable[Dict[str, Any]]) -> List[str]:
    symbols: List[str] = []
    for rule in rules:
        if not isinstance(rule, dict):
            continue
        if norm_text(rule.get("scope") or "domain") == "meta":
            continue
        features = rule.get("match_features") if isinstance(rule.get("match_features"), dict) else {}
        symbols.extend(str(item) for item in (features.get("required_symbols") or []))
    return ordered_unique(symbols)


def build_topic_candidates(catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for domain in catalog.get("domains", []) or []:
        domain_name = norm_text(domain.get("name") or "Unknown")
        for topic in domain.get("topics", []) or []:
            topic_name = norm_text(topic.get("name") or "Unknown")
            topic_obj = dict(topic)
            topic_obj.setdefault("domain", domain_name)
            executable_rule_count = len(topic.get("rules", []) or [])
            tagged_reference = topic.get("tagged_reference") if isinstance(topic.get("tagged_reference"), dict) else {}
            retrieval_hints = topic.get("retrieval_hints") if isinstance(topic.get("retrieval_hints"), dict) else {}
            knowledge_reference = topic.get("knowledge_reference") if isinstance(topic.get("knowledge_reference"), dict) else {}
            out.append(
                {
                    "domain": domain_name,
                    "topic": topic_name,
                    "topic_obj": topic_obj,
                    "aliases": ordered_unique(tagged_reference.get("aliases") or []),
                    "scene_keywords": ordered_unique(retrieval_hints.get("scene_keywords") or []),
                    "topic_keywords": ordered_unique(retrieval_hints.get("topic_keywords") or []),
                    "knowledge_keywords": ordered_unique(knowledge_reference.get("keywords") or []),
                    "required_symbols": ordered_unique(retrieval_hints.get("required_symbols") or []),
                    "executable_rule_count": executable_rule_count,
                }
            )
    return out


def build_signal_document_frequency(topic_candidates: List[Dict[str, Any]]) -> Dict[str, Counter[str]]:
    keyword_df: Counter[str] = Counter()
    scene_df: Counter[str] = Counter()
    symbol_df: Counter[str] = Counter()

    for candidate in topic_candidates:
        for item in ordered_unique(
            list(candidate.get("knowledge_keywords") or []) + list(candidate.get("topic_keywords") or [])
        ):
            keyword_df[item.casefold()] += 1
        for item in ordered_unique(candidate.get("scene_keywords") or []):
            scene_df[item.casefold()] += 1
        for item in ordered_unique(candidate.get("required_symbols") or []):
            symbol_df[item.casefold()] += 1

    return {
        "keyword_df": keyword_df,
        "scene_df": scene_df,
        "symbol_df": symbol_df,
    }


def topic_executable_rule_count(item: Dict[str, Any]) -> int:
    topic_obj = item.get("topic_obj")
    if not isinstance(topic_obj, dict):
        topic_obj = item.get("topic")
    if not isinstance(topic_obj, dict):
        return 0
    return len(topic_obj.get("rules", []) or [])


def _df_weight(value: str, df_counter: Counter[str], *, strong: float, weak: float) -> float:
    df = int(df_counter.get(norm_text(value).casefold(), 0) or 1)
    if df <= 1:
        return strong
    if df <= 3:
        return strong * 0.75
    if df <= 8:
        return (strong + weak) / 2
    return weak


def _symbol_df_weight(symbol: str, df_counter: Counter[str]) -> float:
    normalized = norm_text(symbol)
    if not normalized:
        return 0.0
    df = int(df_counter.get(normalized.casefold(), 0) or 1)
    if len(normalized) == 1 and normalized in GENERIC_SYMBOLS and df > 6:
        return 0.0
    if df <= 2:
        return 1.0
    if df <= 6:
        return 0.5
    return 0.25


def score_topic_candidate(
    candidate: Dict[str, Any],
    text_for_topic: str,
    *,
    signal_df: Dict[str, Counter[str]],
) -> Dict[str, Any]:
    phrase_hits: List[str] = []
    if match_phrase_or_symbol(candidate["topic"], text_for_topic):
        phrase_hits.append(candidate["topic"])
    for alias in candidate.get("aliases") or []:
        if match_phrase_or_symbol(alias, text_for_topic):
            phrase_hits.append(alias)

    scene_hits = [
        kw for kw in ordered_unique(candidate.get("scene_keywords") or []) if match_phrase_or_symbol(kw, text_for_topic)
    ]
    keyword_pool = ordered_unique(list(candidate.get("knowledge_keywords") or []) + list(candidate.get("topic_keywords") or []))
    keyword_hits = []
    for kw in keyword_pool:
        normalized = norm_text(kw).casefold()
        if normalized.isalpha() and len(normalized) <= 2:
            continue
        if normalized in LOW_SIGNAL_KEYWORDS:
            continue
        if signal_df["keyword_df"].get(normalized, 0) > 18 and len(normalized) <= 8:
            continue
        if match_phrase_or_symbol(kw, text_for_topic):
            keyword_hits.append(kw)

    symbol_gate_open = bool(phrase_hits or scene_hits or len(keyword_hits) >= 2)
    symbol_hits = []
    if symbol_gate_open:
        symbol_hits = [
            sym for sym in ordered_unique(candidate.get("required_symbols") or []) if match_phrase_or_symbol(sym, text_for_topic)
        ]

    phrase_score = 5.0 if phrase_hits else 0.0
    scene_score = min(
        sum(_df_weight(hit, signal_df["scene_df"], strong=4.25, weak=1.0) for hit in scene_hits),
        14.0,
    )
    keyword_score = min(
        sum(_df_weight(hit, signal_df["keyword_df"], strong=1.5, weak=0.25) for hit in keyword_hits),
        6.0,
    )
    symbol_score = min(sum(_symbol_df_weight(hit, signal_df["symbol_df"]) for hit in symbol_hits), 3.0)
    anchor_bonus = 0.0
    if scene_hits:
        anchor_bonus += 1.0
        if len(scene_hits) >= 2:
            anchor_bonus += 0.75
    if phrase_hits and scene_hits:
        anchor_bonus += 0.75

    weak_keyword_only = bool(not phrase_hits and not scene_hits and 0 < len(keyword_hits) <= 3)
    weak_keyword_penalty = 0.0
    weak_keyword_context = sum(1 for hit in keyword_hits if _is_physical_context_token(hit))
    if weak_keyword_only:
        if len(keyword_hits) == 1:
            keyword_score *= 0.35
            symbol_score = 0.0
            weak_keyword_penalty = 1.0
        elif len(keyword_hits) == 2:
            keyword_score *= 0.55
            symbol_score = min(symbol_score, 0.5)
            weak_keyword_penalty = 0.75
        else:
            keyword_score *= 0.75
            symbol_score = min(symbol_score, 1.0)
            weak_keyword_penalty = 0.25
        if weak_keyword_context == 0:
            keyword_score *= 0.5
            symbol_score = min(symbol_score, 0.5)
            weak_keyword_penalty += 1.25
        elif weak_keyword_context == 1:
            weak_keyword_penalty += 0.5
    score = float(
        max(0.0, phrase_score + scene_score + keyword_score + symbol_score + anchor_bonus - weak_keyword_penalty)
    )

    return {
        "domain": candidate["domain"],
        "topic": candidate["topic"],
        "score": score,
        "evidence": {
            "name_or_alias_hits": ordered_unique(phrase_hits),
            "scene_keyword_hits": scene_hits,
            "keyword_hits": keyword_hits,
            "required_symbol_hits": symbol_hits,
            "symbol_gate_open": symbol_gate_open,
            "weak_keyword_only": weak_keyword_only,
            "weak_keyword_penalty": weak_keyword_penalty,
            "executable_rule_count": int(candidate.get("executable_rule_count") or 0),
        },
        "topic_obj": candidate["topic_obj"],
    }


def select_topic_matches_with_rule_fallback(
    scored: List[Dict[str, Any]],
    *,
    top_k: int,
    scan_limit: int = 12,
    same_domain_gap: float = 6.0,
    cross_domain_gap: float = 4.0,
) -> List[Dict[str, Any]]:
    if not scored or top_k <= 0:
        return []

    selected = list(scored[:top_k])
    if topic_executable_rule_count(selected[0]) > 0:
        return selected

    top1 = selected[0]
    top1_domain = str(top1.get("domain") or "")
    top1_score = float(top1.get("score") or 0.0)
    same_domain_candidate = None
    cross_domain_candidate = None

    for item in scored[1 : min(len(scored), scan_limit)]:
        if topic_executable_rule_count(item) <= 0:
            continue
        gap = top1_score - float(item.get("score") or 0.0)
        if str(item.get("domain") or "") == top1_domain and gap <= same_domain_gap:
            same_domain_candidate = item
            break
        if cross_domain_candidate is None and gap <= cross_domain_gap:
            cross_domain_candidate = item

    promoted = same_domain_candidate or cross_domain_candidate
    if promoted is None:
        return selected

    def _mark(item: Dict[str, Any], *, promoted_from_empty_topic: bool) -> Dict[str, Any]:
        cloned = dict(item)
        evidence = dict(cloned.get("evidence") or {})
        evidence["promoted_from_empty_topic"] = promoted_from_empty_topic
        evidence["top1_empty_topic_fallback"] = True
        cloned["evidence"] = evidence
        return cloned

    out = [_mark(promoted, promoted_from_empty_topic=True), _mark(top1, promoted_from_empty_topic=False)]
    seen = {
        (str(promoted.get("domain") or ""), str(promoted.get("topic") or promoted.get("name") or "")),
        (str(top1.get("domain") or ""), str(top1.get("topic") or top1.get("name") or "")),
    }
    for item in selected[1:]:
        key = (str(item.get("domain") or ""), str(item.get("topic") or item.get("name") or ""))
        if key in seen:
            continue
        out.append(item)
        seen.add(key)
        if len(out) >= top_k:
            break
    return out[:top_k]


def score_rule_candidate(rule: Dict[str, Any], text_for_rule: str) -> Dict[str, Any]:
    match_features = rule.get("match_features") if isinstance(rule.get("match_features"), dict) else {}
    trigger_hits = [
        kw for kw in ordered_unique(match_features.get("trigger_keywords") or []) if match_phrase_or_symbol(kw, text_for_rule)
    ]
    object_hits = [
        kw for kw in ordered_unique(match_features.get("object_keywords") or []) if match_phrase_or_symbol(kw, text_for_rule)
    ]
    symbol_hits = [
        sym for sym in ordered_unique(match_features.get("required_symbols") or []) if match_phrase_or_symbol(sym, text_for_rule)
    ]
    lexical_hits = len(trigger_hits) + len(object_hits) + len(symbol_hits)

    support = rule.get("support") if isinstance(rule.get("support"), dict) else {}
    count = int(support.get("count") or 0)
    support_prior = min(math.log2(count + 1), 3.0) if lexical_hits > 0 else 0.0
    scope = norm_text(rule.get("scope") or "domain") or "domain"
    manual_override_reason = norm_text(rule.get("manual_override_reason") or "")

    def _is_generic_signal(value: str) -> bool:
        normalized = norm_text(value).casefold()
        if not normalized:
            return False
        if normalized in GENERIC_RULE_SIGNAL_TERMS:
            return True
        return any(term in normalized for term in GENERIC_RULE_SIGNAL_TERMS)

    generic_trigger_hits = [hit for hit in trigger_hits if _is_generic_signal(hit)]
    generic_object_hits = [hit for hit in object_hits if _is_generic_signal(hit)]
    generic_signal_only = bool(lexical_hits > 0 and not set(trigger_hits).difference(generic_trigger_hits) and not set(object_hits).difference(generic_object_hits))

    score = (
        min(len(trigger_hits) * 3, 9)
        + min(len(object_hits) * 2, 6)
        + min(len(symbol_hits) * 2, 4)
        + support_prior
    )
    if scope == "meta":
        score = max(score - 1.5, 0.0)
        if not trigger_hits and not object_hits:
            score = max(score - 0.5, 0.0)

    return {
        "rule_id": norm_text(rule.get("rule_id") or ""),
        "title": norm_text(rule.get("title") or ""),
        "score": float(score),
        "scope": scope,
        "evidence": {
            "trigger_hits": trigger_hits,
            "object_hits": object_hits,
            "required_symbol_hits": symbol_hits,
            "generic_trigger_hits": generic_trigger_hits,
            "generic_object_hits": generic_object_hits,
            "generic_signal_only": generic_signal_only,
            "support_count": count,
            "support_prior": round(support_prior, 4),
            "manual_override_reason": manual_override_reason,
        },
    }


def topic_sort_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    return (-float(item.get("score") or 0.0), str(item.get("domain") or ""), str(item.get("topic") or ""))


def rule_sort_key(item: Dict[str, Any]) -> Tuple[Any, ...]:
    scope_rank = 1 if norm_text(item.get("scope") or "domain") == "meta" else 0
    return (
        scope_rank,
        -float(item.get("adjusted_score", item.get("score") or 0.0)),
        str(item.get("domain") or ""),
        str(item.get("topic") or ""),
        str(item.get("rule_id") or ""),
    )


def rule_topic_context(
    *,
    raw_score: float,
    topic_rank: int,
    topic_score: float,
    top1_topic_score: float,
    scope: str,
    rule_evidence: Dict[str, Any] | None = None,
    topic_evidence: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    gap = max(0.0, float(top1_topic_score or 0.0) - float(topic_score or 0.0))
    evidence = rule_evidence if isinstance(rule_evidence, dict) else {}
    topic_hits = topic_evidence if isinstance(topic_evidence, dict) else {}
    generic_signal_only = bool(evidence.get("generic_signal_only"))
    has_topic_anchor = bool(topic_hits.get("name_or_alias_hits") or topic_hits.get("scene_keyword_hits"))
    normalized_scope = norm_text(scope or "domain") or "domain"

    if topic_rank <= 0:
        min_score = 1.0
        bonus = 1.5
    elif gap <= 0.5:
        min_score = 4.0
        bonus = 0.0
    elif gap <= 1.5:
        min_score = 6.0
        bonus = -0.75
    elif gap <= 3.0:
        min_score = 7.0
        bonus = -1.75
    else:
        min_score = 8.0
        bonus = -2.5

    if normalized_scope == "meta":
        min_score += 0.5
        bonus -= 0.75

    if generic_signal_only:
        min_score += 1.0
        bonus -= 1.0
        if not has_topic_anchor:
            min_score += 2.0
            bonus -= 1.5

    return {
        "topic_gap": gap,
        "min_score": min_score,
        "adjusted_score": float(raw_score + bonus),
    }


def non_top1_rule_quota(*, top1_margin: float, top_n: int) -> int:
    if top1_margin >= 3.0:
        return 0
    if top1_margin > 1.5:
        return min(1, top_n)
    if top1_margin > 0.5:
        return min(2, top_n)
    return top_n


def select_rules_with_topic_priority(
    scored: List[Dict[str, Any]],
    *,
    top_n: int,
    top1_key: Tuple[str, str] | None,
    top1_margin: float,
) -> List[Dict[str, Any]]:
    eligible = [
        item
        for item in scored
        if float(item.get("score") or 0.0) >= float(item.get("min_score") or 0.0)
        and float(item.get("adjusted_score", item.get("score") or 0.0)) > 0.0
    ]
    if not top1_key:
        return eligible[:top_n]

    if top1_margin >= 3.0:
        top1_only = [
            item for item in eligible if (str(item.get("domain") or ""), str(item.get("topic") or "")) == top1_key
        ]
        if top1_only:
            return top1_only[:top_n]
        top1_relaxed = [
            item
            for item in scored
            if (str(item.get("domain") or ""), str(item.get("topic") or "")) == top1_key
            and float(item.get("score") or 0.0) > 0.0
            and float(item.get("adjusted_score", item.get("score") or 0.0)) > 0.0
        ]
        if top1_relaxed:
            return top1_relaxed[:top_n]
        return []

    eligible_top1_count = sum(
        1
        for item in eligible
        if (str(item.get("domain") or ""), str(item.get("topic") or "")) == top1_key
    )
    quota = non_top1_rule_quota(top1_margin=top1_margin, top_n=top_n)

    selected: List[Dict[str, Any]] = []
    non_top1_count = 0
    for item in eligible:
        item_key = (str(item.get("domain") or ""), str(item.get("topic") or ""))
        is_top1 = item_key == top1_key
        if not is_top1 and non_top1_count >= quota and eligible_top1_count >= max(top_n - quota, 0):
            continue
        selected.append(item)
        if not is_top1:
            non_top1_count += 1
        if len(selected) >= top_n:
            break

    return selected
