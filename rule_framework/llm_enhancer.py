"""LLM-based retrieval-signal enhancement for the unified rule catalog.

Adds LLM-generated matching signals that supplement purely keyword-based
retrieval signals produced by the static builder:

Per-rule ``llm_hints``:
  - ``match_phrases``: 3-5 natural-language sentence fragments that would appear
    in a student's solution when this rule is triggered.  Used for fuzzy
    phrase matching in ``score_rule_candidate``.
  - ``discriminative_terms``: 5-8 specific physics terms / short expressions that
    strongly indicate this rule is relevant.

Per-topic ``retrieval_hints`` additions:
  - ``llm_problem_phrases``: 3-5 typical physics problem sentence fragments for
    this topic (used in ``score_topic_candidate``).
  - ``llm_discriminative_terms``: 6-10 topic-specific physics terms.

Semantic cluster enhancement (replaces heuristic ``error_type`` bucketing with
meaningful physics reasoning categories):
  - Each cluster gets ``description`` and ``discriminative_phrases`` for structural
    transparency.

Usage via CLI:
    python scripts/manage_rule_library.py enhance \\
        --catalog <input_catalog.json> --model qwen3-30b-a3b-instruct-2507 \\
        --output <enhanced_catalog.json>
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

try:
    import openai as _openai  # type: ignore
except ImportError:
    _openai = None  # type: ignore

from core.rule_catalog_retrieval import iter_rule_leaves, norm_text, ordered_unique, TOKEN_RE, keep_token


# ── LLM client helpers ─────────────────────────────────────────────────────────

def _make_client() -> Optional[Any]:
    if _openai is None:
        return None
    try:
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
        client = _openai.OpenAI(base_url=base_url)
        if not getattr(client, "api_key", None):
            return None
        return client
    except Exception:
        return None


def _repair_truncated_json(raw: str) -> str:
    """Best-effort repair of a JSON object/array truncated by a token limit."""
    # Try to find the outermost { or [ and balance brackets/braces
    raw = raw.strip()
    start_brace = raw.find("{")
    start_bracket = raw.find("[")
    if start_brace == -1 and start_bracket == -1:
        return raw
    # Use whichever comes first
    if start_brace != -1 and (start_bracket == -1 or start_brace <= start_bracket):
        opener, closer = "{", "}"
        start = start_brace
    else:
        opener, closer = "[", "]"
        start = start_bracket

    body = raw[start:]
    depth = 0
    in_str = False
    escape = False
    last_valid = start
    for i, ch in enumerate(body):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_str:
            escape = True
            continue
        if ch == '"' and not escape:
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                last_valid = i + 1
                break
    # If depth > 0 the JSON was truncated; try to close open structures
    if depth > 0:
        # Drop any trailing incomplete string or incomplete array item
        fragment = body[:last_valid] if last_valid > 0 else body
        # Remove trailing comma + whitespace before we add closers
        fragment = re.sub(r",\s*$", "", fragment.rstrip())
        fragment += closer * depth
        return fragment
    return body[:last_valid]


def _llm_json(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> Any:
    """Call LLM and parse the first JSON object/array from the response."""
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = str(resp.choices[0].message.content or "")
    # Try direct parse first
    try:
        return json.loads(raw.strip())
    except Exception:
        pass
    # Try repaired parse
    repaired = _repair_truncated_json(raw)
    try:
        return json.loads(repaired)
    except Exception:
        pass
    return {}


def _default_precision_metadata(rule: Dict[str, Any]) -> Dict[str, Any]:
    """Build conservative precision metadata when LLM output is absent or partial."""
    features = rule.get("match_features") if isinstance(rule.get("match_features"), dict) else {}
    required_symbols = [norm_text(x) for x in (features.get("required_symbols") or []) if norm_text(x)]
    trigger_keywords = [norm_text(x) for x in (features.get("trigger_keywords") or []) if norm_text(x)]
    object_keywords = [norm_text(x) for x in (features.get("object_keywords") or []) if norm_text(x)]
    symbolic_hint = rule.get("symbolic_hint") if isinstance(rule.get("symbolic_hint"), dict) else {}
    symbolic_policy = "suppress_on_inconclusive" if symbolic_hint.get("primitive") not in {"", None, "none"} else "suppress_on_pass"
    profile = "balanced" if norm_text(rule.get("scope") or "") == "meta" else "strict"
    return {
        "precision_profile": profile,
        "publishable": True,
        "preconditions": ordered_unique(trigger_keywords[:3] + object_keywords[:2]),
        "violation_signatures": ordered_unique(required_symbols[:3] + trigger_keywords[:2]),
        "negative_conditions": [
            "intermediate value later corrected",
            "rule not required by the problem",
            "equivalent formulation",
        ],
        "evidence_requirements": ordered_unique(required_symbols[:3] + trigger_keywords[:2]),
        "symbolic_policy": symbolic_policy,
    }


def _clean_precision_metadata(raw: Any, rule: Dict[str, Any]) -> Dict[str, Any]:
    base = _default_precision_metadata(rule)
    if not isinstance(raw, dict):
        return base

    profile = norm_text(raw.get("precision_profile") or raw.get("profile") or base["precision_profile"]).lower()
    if profile not in {"strict", "balanced", "recall"}:
        profile = base["precision_profile"]
    symbolic_policy = norm_text(raw.get("symbolic_policy") or base["symbolic_policy"]).lower()
    if symbolic_policy not in {"suppress_on_pass", "suppress_on_inconclusive", "require_fail"}:
        symbolic_policy = base["symbolic_policy"]
    publishable_raw = raw.get("publishable", base["publishable"])
    if isinstance(publishable_raw, str):
        publishable = publishable_raw.strip().lower() not in {"0", "false", "no", "off"}
    else:
        publishable = bool(publishable_raw)

    def _items(key: str, limit: int) -> List[str]:
        values = [norm_text(x) for x in (raw.get(key) or []) if norm_text(x)]
        return ordered_unique(values)[:limit] or list(base[key])

    return {
        "precision_profile": profile,
        "publishable": publishable,
        "preconditions": _items("preconditions", 6),
        "violation_signatures": _items("violation_signatures", 6),
        "negative_conditions": _items("negative_conditions", 6),
        "evidence_requirements": _items("evidence_requirements", 6),
        "symbolic_policy": symbolic_policy,
    }


def _attach_precision_metadata(rule: Dict[str, Any], raw: Any = None) -> None:
    metadata = _clean_precision_metadata(raw, rule)
    rule["precision_profile"] = metadata["precision_profile"]
    rule["publishable"] = metadata["publishable"]
    rule["preconditions"] = metadata["preconditions"]
    rule["violation_signatures"] = metadata["violation_signatures"]
    rule["negative_conditions"] = metadata["negative_conditions"]
    rule["evidence_requirements"] = metadata["evidence_requirements"]
    rule["symbolic_policy"] = metadata["symbolic_policy"]


# ── Per-rule LLM hint generation ───────────────────────────────────────────────

_RULE_HINT_SYSTEM = (
    "You are a physics education expert building a rule-matching system. "
    "Respond with valid JSON only. No markdown fences."
)

_RULE_HINT_USER = """\
Given this physics verification rule:
- Domain/Topic: {domain} / {topic}
- Title: {title}
- Trigger: {trigger}
- Check Logic: {check_logic}

Generate retrieval hints so the system can detect when a student solution violates
this rule.  Return JSON exactly:
{{
  "match_phrases": [
    "phrase 1 (10-30 words appearing in a student solution that triggers this rule)",
    "phrase 2",
    "phrase 3"
  ],
  "discriminative_terms": [
    "specific term 1 (1-4 words, highly specific to this rule scenario)",
    "term 2",
    "term 3",
    "term 4",
    "term 5"
  ],
  "precision_profile": "strict",
  "preconditions": [
    "specific condition that must be present before this rule applies"
  ],
  "violation_signatures": [
    "direct formula, unit, value-range, or reasoning pattern proving violation"
  ],
  "negative_conditions": [
    "context where this rule must NOT trigger"
  ],
  "evidence_requirements": [
    "tokens or phrases that must appear in the quoted evidence"
  ],
  "symbolic_policy": "suppress_on_pass"
}}

Constraints:
- match_phrases must read like real student solution text, not rule descriptions.
- discriminative_terms must be physics-specific and distinguish this rule from similar rules.
  Bad: "energy", "force", "equation".  Good: "Carnot efficiency", "reduced mass", "gyroscopic precession".
- precision_profile must be "strict" unless the rule is inherently broad; never use "recall" for final-answer checks.
- preconditions and violation_signatures must be concrete, not generic descriptions.
- symbolic_policy is one of "suppress_on_pass", "suppress_on_inconclusive", or "require_fail".
- Respond ONLY with the JSON object."""


def _enhance_rule_leaf(
    rule: Dict[str, Any],
    *,
    domain: str,
    topic: str,
    client: Any,
    model: str,
    sleep_sec: float = 0.0,
) -> Dict[str, Any]:
    """Return a copy of *rule* augmented with ``llm_hints`` (single-rule call)."""
    title = norm_text(rule.get("title") or "")
    trigger = norm_text(rule.get("trigger") or "")
    check_logic = norm_text(rule.get("check_logic") or "")
    if not title and not trigger:
        return rule

    user = _RULE_HINT_USER.format(
        domain=domain[:80],
        topic=topic[:80],
        title=title[:200],
        trigger=trigger[:350],
        check_logic=check_logic[:350],
    )
    try:
        result = _llm_json(client, model, _RULE_HINT_SYSTEM, user, max_tokens=550)
        if sleep_sec:
            time.sleep(sleep_sec)
    except Exception:
        return rule

    if not isinstance(result, dict):
        return rule

    phrases = [norm_text(p) for p in (result.get("match_phrases") or []) if norm_text(p)]
    terms = [norm_text(t) for t in (result.get("discriminative_terms") or []) if norm_text(t)]
    patched = dict(rule)
    patched["llm_hints"] = {
        "match_phrases": ordered_unique(phrases)[:5],
        "discriminative_terms": ordered_unique(terms)[:8],
    }
    _attach_precision_metadata(patched, result)
    return patched


# ── Batched rule hint generation (reduces LLM call count) ─────────────────────

_BATCH_RULE_SYSTEM = (
    "You are a physics education expert building a rule-matching system. "
    "Respond with valid JSON only. No markdown fences."
)

_BATCH_RULE_USER = """\
Domain/Topic: {domain} / {topic}

Below are {n} physics verification rules.  For each rule, generate retrieval hints.

Rules:
{rules_section}

Return JSON exactly — a list with one entry per rule in the SAME ORDER:
[
  {{
    "rule_id": "...",
    "match_phrases": [
      "phrase (10-30 words from student solution when this rule triggers)",
      "phrase 2"
    ],
    "discriminative_terms": [
      "specific term 1 (1-4 words, highly specific to this rule)",
      "term 2",
      "term 3"
    ],
    "precision_profile": "strict",
    "preconditions": ["concrete applicability condition"],
    "violation_signatures": ["direct evidence pattern for the violation"],
    "negative_conditions": ["context where the rule must not trigger"],
    "evidence_requirements": ["required token or phrase in quoted evidence"],
    "symbolic_policy": "suppress_on_pass"
  }},
  ...
]

Constraints:
- match_phrases must read like real student solution text.
- discriminative_terms must be physics-specific (e.g. "reduced mass", "Carnot efficiency").
  Avoid generic: "energy", "force", "equation".
- precision fields are used as final-publication gates. Make them conservative and concrete.
- symbolic_policy is one of "suppress_on_pass", "suppress_on_inconclusive", or "require_fail".
- Return exactly {n} entries in the same order as the input rules.
- Respond ONLY with the JSON array."""


def _enhance_rules_batch(
    rules: List[Dict[str, Any]],
    *,
    domain: str,
    topic: str,
    client: Any,
    model: str,
    sleep_sec: float = 0.0,
) -> None:
    """In-place: add ``llm_hints`` to each rule in *rules* via a single batched LLM call."""
    if not rules:
        return

    rules_section_parts: List[str] = []
    for i, rule in enumerate(rules):
        rid = norm_text(rule.get("rule_id") or rule.get("id") or f"rule_{i}")
        title = norm_text(rule.get("title") or "")[:150]
        trigger = norm_text(rule.get("trigger") or "")[:200]
        check_logic = norm_text(rule.get("check_logic") or "")[:200]
        rules_section_parts.append(
            f"[{i+1}] rule_id={rid}\n  Title: {title}\n  Trigger: {trigger}\n  Check Logic: {check_logic}"
        )
    rules_section = "\n\n".join(rules_section_parts)

    user = _BATCH_RULE_USER.format(
        domain=domain[:80],
        topic=topic[:80],
        n=len(rules),
        rules_section=rules_section,
    )

    try:
        result = _llm_json(client, model, _BATCH_RULE_SYSTEM, user, max_tokens=min(200 * len(rules), 2048))
        if sleep_sec:
            time.sleep(sleep_sec)
    except Exception:
        return

    if not isinstance(result, list):
        return

    # Match returned entries back to rules by position (or rule_id fallback)
    id_to_rule = {norm_text(r.get("rule_id") or r.get("id") or ""): r for r in rules}
    for idx, entry in enumerate(result):
        if not isinstance(entry, dict):
            continue
        # Try match by returned rule_id first, then by position
        rid = norm_text(entry.get("rule_id") or "")
        rule = id_to_rule.get(rid) if rid and rid in id_to_rule else (rules[idx] if idx < len(rules) else None)
        if rule is None:
            continue
        phrases = [norm_text(p) for p in (entry.get("match_phrases") or []) if norm_text(p)]
        terms = [norm_text(t) for t in (entry.get("discriminative_terms") or []) if norm_text(t)]
        rule["llm_hints"] = {
            "match_phrases": ordered_unique(phrases)[:5],
            "discriminative_terms": ordered_unique(terms)[:6],
        }
        _attach_precision_metadata(rule, entry)


# ── Per-topic LLM hint generation ──────────────────────────────────────────────

_TOPIC_HINT_SYSTEM = (
    "You are a physics education expert building a rule-matching system. "
    "Respond with valid JSON only. No markdown fences."
)

_TOPIC_HINT_USER = """\
Physics catalog topic: {domain} / {topic}
Number of rules: {n_rules}
Sample rule titles: {sample_titles}

Generate retrieval hints so the system identifies physics competition problems
belonging to this topic.  Return JSON exactly:
{{
  "problem_phrases": [
    "phrase 1 (15-45 words from a physics problem statement specific to this topic)",
    "phrase 2",
    "phrase 3"
  ],
  "discriminative_terms": [
    "specific term 1 (1-4 words, clearly indicates this topic vs similar ones)",
    "term 2",
    "term 3",
    "term 4",
    "term 5",
    "term 6"
  ]
}}

Constraints:
- problem_phrases should sound like real IPhO/physics-olympiad problem text.
- discriminative_terms must distinguish this topic from neighboring topics.
  E.g. for "Circular Motion" → good: "centripetal acceleration", "conical pendulum",
  bad: "motion", "velocity", "force".
- Respond ONLY with the JSON object."""


def _enhance_topic_hints(
    topic_entry: Dict[str, Any],
    *,
    domain: str,
    client: Any,
    model: str,
    sleep_sec: float = 0.0,
) -> None:
    """In-place: add LLM retrieval phrases to ``topic_entry["retrieval_hints"]``."""
    topic_name = norm_text(topic_entry.get("name") or "")
    rules = list(iter_rule_leaves(topic_entry))
    if not rules:
        return

    sample_titles = [norm_text(r.get("title") or "") for r in rules[:8] if norm_text(r.get("title") or "")]
    sample_titles_str = "; ".join(sample_titles[:5]) or "(none)"

    user = _TOPIC_HINT_USER.format(
        domain=domain[:80],
        topic=topic_name[:80],
        n_rules=len(rules),
        sample_titles=sample_titles_str[:400],
    )
    try:
        result = _llm_json(client, model, _TOPIC_HINT_SYSTEM, user, max_tokens=550)
        if sleep_sec:
            time.sleep(sleep_sec)
    except Exception:
        return

    if not isinstance(result, dict):
        return

    phrases = [norm_text(p) for p in (result.get("problem_phrases") or []) if norm_text(p)]
    terms = [norm_text(t) for t in (result.get("discriminative_terms") or []) if norm_text(t)]
    hints = topic_entry.setdefault("retrieval_hints", {})
    hints["llm_problem_phrases"] = ordered_unique(phrases)[:5]
    hints["llm_discriminative_terms"] = ordered_unique(terms)[:10]


# ── Semantic cluster enhancement ───────────────────────────────────────────────

_CLUSTER_SYSTEM = (
    "You are a physics education expert organising verification rules into clusters. "
    "Respond with valid JSON only. No markdown fences."
)

_CLUSTER_USER = """\
Topic: {domain} / {topic}
Rules ({n_rules} total):
{rules_summary}

Group these rules into 2-5 semantically meaningful clusters based on:
1. The TYPE of physical reasoning error (sign error, wrong formula, missing term, etc.)
2. The PHYSICAL PHENOMENON involved (energy conservation, force balance, wave optics, etc.)
3. The STEP in the solution where the error occurs (setup, equation, calculation, interpretation)

Return JSON exactly:
{{
  "clusters": [
    {{
      "label": "Concise cluster name (3-7 words)",
      "description": "What characterises errors in this cluster (10-25 words)",
      "discriminative_phrases": [
        "Short phrase (5-20 words) found in solutions with these errors"
      ],
      "rule_ids": ["rule_id_1", "rule_id_2"]
    }}
  ]
}}

Constraints:
- Every rule_id MUST appear in exactly ONE cluster.
- Cluster labels must be specific (e.g. "Incorrect Force Decomposition in Rotating Frame")
  not generic (e.g. "Logic Errors").
- discriminative_phrases (1-3 per cluster) should appear verbatim in erroneous solutions.
- Respond ONLY with the JSON object."""


def _enhance_clusters_for_topic(
    topic_entry: Dict[str, Any],
    *,
    domain: str,
    client: Any,
    model: str,
    min_rules: int = 4,
    sleep_sec: float = 0.0,
) -> None:
    """In-place: replace heuristic clusters with LLM semantic clusters."""
    topic_name = norm_text(topic_entry.get("name") or "")
    rules = list(iter_rule_leaves(topic_entry))
    if len(rules) < min_rules:
        return

    rules_summary_lines: List[str] = []
    for r in rules[:24]:
        rid = norm_text(r.get("rule_id") or r.get("id") or "")
        title = norm_text(r.get("title") or "")
        trigger = norm_text(r.get("trigger") or "")[:80]
        rules_summary_lines.append(f"- {rid}: {title} | {trigger}")
    rules_summary = "\n".join(rules_summary_lines)

    user = _CLUSTER_USER.format(
        domain=domain[:80],
        topic=topic_name[:80],
        n_rules=len(rules),
        rules_summary=rules_summary,
    )
    try:
        result = _llm_json(client, model, _CLUSTER_SYSTEM, user, max_tokens=1000)
        if sleep_sec:
            time.sleep(sleep_sec)
    except Exception:
        return

    if not isinstance(result, dict):
        return

    raw_clusters = result.get("clusters") or []
    if not isinstance(raw_clusters, list) or not raw_clusters:
        return

    valid_ids = {norm_text(r.get("rule_id") or r.get("id") or "") for r in rules}
    new_clusters: List[Dict[str, Any]] = []
    assigned_ids: set = set()

    for raw_c in raw_clusters:
        if not isinstance(raw_c, dict):
            continue
        label = norm_text(raw_c.get("label") or "")
        desc = norm_text(raw_c.get("description") or "")
        disc = [norm_text(p) for p in (raw_c.get("discriminative_phrases") or []) if norm_text(p)]
        rids = [norm_text(rid) for rid in (raw_c.get("rule_ids") or [])
                if norm_text(rid) in valid_ids and norm_text(rid) not in assigned_ids]
        if not label or not rids:
            continue
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower())[:32].strip("_")
        new_clusters.append({
            "cluster_id": f"cluster_{slug}",
            "label": label,
            "description": desc,
            "discriminative_phrases": ordered_unique(disc)[:3],
            "rule_ids": rids,
        })
        assigned_ids.update(rids)

    # Gather any rule_ids the LLM missed and attach to last cluster
    unassigned = [norm_text(r.get("rule_id") or r.get("id") or "")
                  for r in rules if norm_text(r.get("rule_id") or r.get("id") or "") not in assigned_ids]
    if unassigned:
        if new_clusters:
            new_clusters[-1]["rule_ids"].extend(unassigned)
        else:
            new_clusters.append({
                "cluster_id": "cluster_general",
                "label": "General Physics Errors",
                "description": "Miscellaneous physics reasoning errors",
                "discriminative_phrases": [],
                "rule_ids": unassigned,
            })

    if not new_clusters:
        return

    topic_entry["clusters"] = new_clusters

    # Propagate cluster_id back to rule path metadata
    cluster_by_rule: Dict[str, str] = {}
    for cluster in new_clusters:
        for rid in cluster.get("rule_ids") or []:
            cluster_by_rule[rid] = str(cluster.get("cluster_id") or "unclustered")
    for rule in rules:
        rid = norm_text(rule.get("rule_id") or rule.get("id") or "")
        path = dict(rule.get("path") or {})
        path["cluster"] = cluster_by_rule.get(rid, "unclustered")
        rule["path"] = path


# ── Top-level orchestrator ─────────────────────────────────────────────────────

def enhance_catalog(
    catalog: Dict[str, Any],
    *,
    model: str,
    do_rule_hints: bool = True,
    do_topic_hints: bool = True,
    do_semantic_clusters: bool = True,
    cluster_min_rules: int = 4,
    rule_batch_size: int = 6,
    sleep_between_calls: float = 0.0,
    refresh_existing: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Add LLM-generated retrieval signals in-place and return the catalog.

    Phases (each optional):
    1. Per-topic LLM retrieval hints (``llm_problem_phrases``,
       ``llm_discriminative_terms`` inside ``retrieval_hints``).
    2. Per-rule LLM hints (``llm_hints`` containing ``match_phrases`` and
       ``discriminative_terms`` on each rule leaf), using *rule_batch_size* rules
       per LLM call to reduce total call count (~514 rules / 6 = ~86 calls vs 514).
    3. LLM semantic clustering replacing heuristic ``error_type`` buckets.
    """
    client = _make_client()
    if client is None:
        raise RuntimeError(
            "Cannot initialise OpenAI client. "
            "Ensure OPENAI_API_KEY (and OPENAI_BASE_URL if needed) are set."
        )

    enhanced_topics = 0
    enhanced_rules = 0
    skipped_rule_hints = 0
    skipped_topic_hints = 0
    failed_calls = 0

    for domain_obj in catalog.get("domains", []) or []:
        domain_name = norm_text(domain_obj.get("name") or "Unknown")
        for topic_entry in domain_obj.get("topics", []) or []:
            topic_name = norm_text(topic_entry.get("name") or "Unknown")
            rules = list(iter_rule_leaves(topic_entry))
            if not rules:
                continue

            if verbose:
                print(f"  [{domain_name}] {topic_name} ({len(rules)} rules)", flush=True)

            # ── Phase 1: topic retrieval hints ─────────────────────────────────
            if do_topic_hints:
                hints = topic_entry.get("retrieval_hints") if isinstance(topic_entry.get("retrieval_hints"), dict) else {}
                has_topic_hints = bool(
                    hints.get("llm_problem_phrases") and hints.get("llm_discriminative_terms")
                )
                if has_topic_hints and not refresh_existing:
                    skipped_topic_hints += 1
                else:
                    try:
                        _enhance_topic_hints(
                            topic_entry,
                            domain=domain_name,
                            client=client,
                            model=model,
                            sleep_sec=sleep_between_calls,
                        )
                    except Exception as exc:
                        failed_calls += 1
                        if verbose:
                            print(f"    [WARN] topic hint: {exc}")

            # ── Phase 2: batched per-rule hints ────────────────────────────────
            if do_rule_hints:
                rule_list = [
                    rule for rule in (topic_entry.get("rules") or [])
                    if refresh_existing or not isinstance(rule.get("llm_hints"), dict)
                ]
                skipped_rule_hints += max(0, len(topic_entry.get("rules") or []) - len(rule_list))
                batch_size = max(1, int(rule_batch_size))
                for batch_start in range(0, len(rule_list), batch_size):
                    batch = rule_list[batch_start: batch_start + batch_size]
                    try:
                        _enhance_rules_batch(
                            batch,
                            domain=domain_name,
                            topic=topic_name,
                            client=client,
                            model=model,
                            sleep_sec=sleep_between_calls,
                        )
                        enhanced_rules += len(batch)
                    except Exception as exc:
                        failed_calls += 1
                        if verbose:
                            print(f"    [WARN] batch rule hint (offset {batch_start}): {exc}")

            # ── Phase 3: semantic clusters ─────────────────────────────────────
            if do_semantic_clusters and len(rules) >= cluster_min_rules:
                try:
                    _enhance_clusters_for_topic(
                        topic_entry,
                        domain=domain_name,
                        client=client,
                        model=model,
                        min_rules=cluster_min_rules,
                        sleep_sec=sleep_between_calls,
                    )
                except Exception as exc:
                    failed_calls += 1
                    if verbose:
                        print(f"    [WARN] cluster enhancement: {exc}")

            for rule in rules:
                if not rule.get("preconditions") or not rule.get("violation_signatures"):
                    _attach_precision_metadata(rule, rule)

            enhanced_topics += 1

    meta = catalog.setdefault("metadata", {})
    meta["llm_enhanced"] = True
    meta["llm_enhance_model"] = model
    meta["llm_enhance_phases"] = {
        "rule_hints": do_rule_hints,
        "topic_hints": do_topic_hints,
        "semantic_clusters": do_semantic_clusters,
    }
    meta["llm_enhance_refresh_existing"] = bool(refresh_existing)
    meta["llm_enhance_coverage"] = summarize_enhancement_coverage(catalog)
    meta["precision_schema"] = {
        "fields": [
            "precision_profile",
            "publishable",
            "preconditions",
            "violation_signatures",
            "negative_conditions",
            "evidence_requirements",
            "symbolic_policy",
        ],
        "default_mode": "strict",
    }

    if verbose:
        print(
            f"\nEnhancement complete: {enhanced_topics} topics, "
            f"{enhanced_rules} rules enhanced, "
            f"{skipped_rule_hints} existing rule hints skipped, "
            f"{skipped_topic_hints} existing topic hints skipped, "
            f"{failed_calls} failed LLM calls."
        )

    return catalog


def summarize_enhancement_coverage(catalog: Dict[str, Any]) -> Dict[str, int]:
    """Return coarse coverage counts for LLM-generated catalog signals."""
    topics_with_rules = 0
    rules_total = 0
    topics_with_llm_hints = 0
    rules_with_llm_hints = 0
    rules_with_precision_metadata = 0
    topics_with_clusters = 0

    for domain_obj in catalog.get("domains", []) or []:
        for topic_entry in domain_obj.get("topics", []) or []:
            rules = list(iter_rule_leaves(topic_entry))
            if not rules:
                continue
            topics_with_rules += 1
            rules_total += len(rules)
            hints = topic_entry.get("retrieval_hints") if isinstance(topic_entry.get("retrieval_hints"), dict) else {}
            if hints.get("llm_problem_phrases") or hints.get("llm_discriminative_terms"):
                topics_with_llm_hints += 1
            if topic_entry.get("clusters"):
                topics_with_clusters += 1
            for rule in rules:
                if isinstance(rule.get("llm_hints"), dict):
                    rules_with_llm_hints += 1
                if rule.get("preconditions") and rule.get("violation_signatures"):
                    rules_with_precision_metadata += 1

    return {
        "topics_with_rules": topics_with_rules,
        "rules_total": rules_total,
        "topics_with_llm_hints": topics_with_llm_hints,
        "rules_with_llm_hints": rules_with_llm_hints,
        "rules_with_precision_metadata": rules_with_precision_metadata,
        "topics_with_clusters": topics_with_clusters,
    }
