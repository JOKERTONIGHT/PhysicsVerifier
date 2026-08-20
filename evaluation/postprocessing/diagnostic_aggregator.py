"""Cluster and merge overlapping diagnostic candidates before publication."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


def _tokenize(text: str) -> Set[str]:
    return {t for t in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(t) >= 3}


def _norm_quote(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().casefold())


def _para_index(diagnostic: Dict[str, Any]) -> int:
    evidence = diagnostic.get("evidence") if isinstance(diagnostic.get("evidence"), dict) else {}
    loc = evidence.get("location") if isinstance(evidence.get("location"), dict) else {}
    try:
        return int(loc.get("paragraph_index") or -1)
    except Exception:
        return -1


def _quote_text(diagnostic: Dict[str, Any]) -> str:
    evidence = diagnostic.get("evidence") if isinstance(diagnostic.get("evidence"), dict) else {}
    return str(evidence.get("quote") or "").strip()


def _message_text(diagnostic: Dict[str, Any]) -> str:
    return str(diagnostic.get("message") or "").strip()


def _rule_score(diagnostic: Dict[str, Any]) -> float:
    gate = diagnostic.get("release_gate") if isinstance(diagnostic.get("release_gate"), dict) else {}
    match = diagnostic.get("rule_match") if isinstance(diagnostic.get("rule_match"), dict) else {}
    for src in (gate, match):
        try:
            return float(src.get("rule_score") or src.get("score") or 0.0)
        except Exception:
            continue
    return 0.0


def _symbolic_rank(diagnostic: Dict[str, Any]) -> int:
    gate = diagnostic.get("release_gate") if isinstance(diagnostic.get("release_gate"), dict) else {}
    status = str(gate.get("symbolic_status") or "none").lower()
    return {"supported": 0, "quote_overlap": 1, "none": 2, "inconclusive": 3}.get(status, 4)


def _specificity_score(diagnostic: Dict[str, Any]) -> Tuple[int, int, float, int]:
    msg = _message_text(diagnostic)
    quote = _quote_text(diagnostic)
    rule = str(diagnostic.get("rule") or "")
    msg_tokens = len(_tokenize(msg))
    quote_tokens = len(_tokenize(quote))
    broad_penalty = 1 if rule.startswith("norm_") and msg_tokens < 12 else 0
    consequence_penalty = 1 if _looks_like_consequence_only(msg) else 0
    return (
        broad_penalty + consequence_penalty,
        -msg_tokens,
        -_rule_score(diagnostic),
        -quote_tokens,
    )


def _looks_like_consequence_only(message: str) -> bool:
    text = str(message or "").lower()
    markers = (
        "final answer",
        "numerical result",
        "wrong value",
        "incorrect result",
        "does not match",
        "off by a factor",
        "arithmetic mistake",
    )
    root_markers = ("because", "since", "violates", "should be", "must use", "incorrect formula", "wrong sign")
    if any(m in text for m in markers) and not any(m in text for m in root_markers):
        return True
    return False


def _quote_overlap_ratio(a: str, b: str) -> float:
    ta = _tokenize(a)
    tb = _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(1, min(len(ta), len(tb)))


def _cluster_key(diagnostic: Dict[str, Any]) -> Tuple[int, str]:
    quote = _quote_text(diagnostic)
    norm = _norm_quote(quote)
    prefix = norm[:96] if norm else ""
    return (_para_index(diagnostic), prefix)


def _same_cluster(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    para_a, prefix_a = _cluster_key(a)
    para_b, prefix_b = _cluster_key(b)
    if para_a >= 1 and para_b >= 1 and para_a == para_b:
        qa = _quote_text(a)
        qb = _quote_text(b)
        if _quote_overlap_ratio(qa, qb) >= 0.55:
            return True
        if prefix_a and prefix_b and (prefix_a == prefix_b or prefix_a.startswith(prefix_b) or prefix_b.startswith(prefix_a)):
            return True
        if _quote_overlap_ratio(_message_text(a), _message_text(b)) >= 0.65:
            return True
    qa = _quote_text(a)
    qb = _quote_text(b)
    if qa and qb and _norm_quote(qa) == _norm_quote(qb):
        return True
    return False


class DiagnosticAggregator:
    """Merge semantically overlapping diagnostics into one concrete finding."""

    def aggregate(
        self,
        diagnostics: List[Dict[str, Any]],
        *,
        rule_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        _ = rule_records
        items = [d for d in diagnostics or [] if isinstance(d, dict)]
        if len(items) <= 1:
            return items, []

        clusters: List[List[Dict[str, Any]]] = []
        for d in sorted(items, key=lambda x: (_symbolic_rank(x), _specificity_score(x))):
            placed = False
            for cluster in clusters:
                if any(_same_cluster(d, other) for other in cluster):
                    cluster.append(d)
                    placed = True
                    break
            if not placed:
                clusters.append([d])

        kept: List[Dict[str, Any]] = []
        suppressed: List[Dict[str, Any]] = []
        for cluster in clusters:
            best = sorted(
                cluster,
                key=lambda x: (_symbolic_rank(x), _specificity_score(x), str(x.get("rule") or "")),
            )[0]
            merged_rules = sorted({str(x.get("rule") or "") for x in cluster if str(x.get("rule") or "")})
            if len(cluster) > 1:
                enriched = dict(best)
                enriched["aggregation"] = {
                    "merged_rule_ids": merged_rules,
                    "merged_count": len(cluster),
                    "kept_rule": str(best.get("rule") or ""),
                }
                kept.append(enriched)
                for other in cluster:
                    if other is best:
                        continue
                    suppressed.append(
                        {
                            "reason": "diagnostic_aggregator_duplicate_cluster",
                            "kept_rule": str(best.get("rule") or ""),
                            "merged_rule_ids": merged_rules,
                            "original_diagnostic": other,
                        }
                    )
            else:
                kept.append(best)
        return kept, suppressed
