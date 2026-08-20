"""Second-stage diagnostic validator to suppress broad or irrelevant findings."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


GENERIC_MESSAGE_MARKERS = (
    "incorrect",
    "wrong",
    "invalid",
    "does not follow",
    "violates",
    "missing",
    "not mentioned",
    "no mention",
    "should consider",
    "fails to",
)


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower()) if len(t) >= 3]


def _quote_text(diagnostic: Dict[str, Any]) -> str:
    evidence = diagnostic.get("evidence") if isinstance(diagnostic.get("evidence"), dict) else {}
    return str(evidence.get("quote") or "").strip()


def _message_text(diagnostic: Dict[str, Any]) -> str:
    return str(diagnostic.get("message") or "").strip()


def _rule_precision(rule_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(rule_record, dict):
        return {}
    rule = rule_record.get("rule") if isinstance(rule_record.get("rule"), dict) else {}
    precision = rule.get("precision") if isinstance(rule.get("precision"), dict) else {}
    return precision if isinstance(precision, dict) else {}


class DiagnosticValidator:
    """Heuristic + optional LLM validator for candidate diagnostics."""

    def __init__(
        self,
        *,
        llm_model: Optional[str] = None,
        use_llm: bool = False,
        temperature: float = 0.0,
    ) -> None:
        self.llm_model = str(llm_model or os.getenv("PHYSICSVERIFIER_VALIDATOR_MODEL") or "").strip()
        self.use_llm = bool(use_llm)
        self.temperature = float(temperature)
        self._client: Any = None

    def _llm_available(self) -> bool:
        return bool(self.use_llm and self.llm_model and OpenAI and os.getenv("OPENAI_API_KEY"))

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if not OpenAI:
            raise RuntimeError("openai package is not installed")
        kwargs: Dict[str, Any] = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE"),
        }
        if httpx is not None:
            kwargs["http_client"] = httpx.Client(trust_env=False)
        self._client = OpenAI(**kwargs)
        return self._client

    def _heuristic_validate(
        self,
        *,
        question: str,
        reference_answer: str,
        student_solution: str,
        diagnostic: Dict[str, Any],
        rule_record: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        msg = _message_text(diagnostic)
        quote = _quote_text(diagnostic)
        rule_id = str(diagnostic.get("rule") or "")
        precision = _rule_precision(rule_record)

        reasons: List[str] = []
        verdict = "valid"

        if not quote:
            return {"verdict": "irrelevant", "reasons": ["missing_quote"], "rewritten_message": None}

        msg_tokens = _tokenize(msg)
        quote_tokens = set(_tokenize(quote))
        overlap = len(set(msg_tokens) & quote_tokens)

        if len(msg) < 24:
            verdict = "too_broad"
            reasons.append("message_too_short")
        elif len(msg_tokens) <= 4 and any(m in msg.lower() for m in GENERIC_MESSAGE_MARKERS):
            verdict = "too_broad"
            reasons.append("generic_short_message")

        negative_conditions = [str(x) for x in (precision.get("negative_conditions") or []) if str(x).strip()]
        for neg in negative_conditions:
            if neg and neg.lower() in quote.lower():
                verdict = "irrelevant"
                reasons.append("quote_hits_negative_condition")
                break

        if rule_id.startswith("norm_") and overlap < 1 and len(msg_tokens) < 10:
            if verdict == "valid":
                verdict = "too_broad"
            reasons.append("broad_norm_rule_low_quote_overlap")

        consequence_markers = ("final answer", "numerical result", "wrong value", "off by")
        root_markers = ("because", "violates", "should be", "must use", "incorrect formula", "wrong sign")
        if any(m in msg.lower() for m in consequence_markers) and not any(m in msg.lower() for m in root_markers):
            if verdict == "valid":
                verdict = "too_broad"
            reasons.append("consequence_not_root_cause")

        if overlap >= 1 and len(msg_tokens) >= 8 and verdict == "valid":
            required = [str(x) for x in (precision.get("evidence_requirements") or []) if str(x).strip()]
            if required and not any(req.lower() in msg.lower() for req in required):
                verdict = "needs_rewrite"
                reasons.append("missing_required_evidence_in_message")

        rewritten = None
        if verdict == "needs_rewrite":
            rewritten = self._rewrite_message(
                diagnostic=diagnostic,
                rule_record=rule_record,
                question=question,
                reference_answer=reference_answer,
            )

        return {
            "verdict": verdict,
            "reasons": reasons,
            "rewritten_message": rewritten,
            "quote_overlap_tokens": overlap,
            "context_chars": {
                "question": len(question or ""),
                "reference_answer": len(reference_answer or ""),
                "student_solution": len(student_solution or ""),
            },
        }

    def _rewrite_message(
        self,
        *,
        diagnostic: Dict[str, Any],
        rule_record: Optional[Dict[str, Any]],
        question: str,
        reference_answer: str,
    ) -> Optional[str]:
        msg = _message_text(diagnostic)
        quote = _quote_text(diagnostic)
        rule = rule_record.get("rule") if isinstance(rule_record, dict) and isinstance(rule_record.get("rule"), dict) else {}
        title = str(rule.get("title") or rule.get("name") or diagnostic.get("rule") or "physics rule")
        signatures = rule.get("violation_signatures") or (rule.get("precision") or {}).get("violation_signatures") or []
        sig = str(signatures[0]) if isinstance(signatures, list) and signatures else ""
        if sig:
            body = f"{sig.rstrip('.')}: \"{quote[:120]}\"."
        else:
            body = f"{title}: the quoted step \"{quote[:120]}\" violates the expected physical relation."
        if len(body) < len(msg):
            return msg
        return body[:480]

    def _llm_validate(
        self,
        *,
        question: str,
        reference_answer: str,
        student_solution: str,
        diagnostic: Dict[str, Any],
        rule_record: Optional[Dict[str, Any]],
        heuristic: Dict[str, Any],
    ) -> Dict[str, Any]:
        if heuristic.get("verdict") in {"too_broad", "irrelevant", "duplicate"}:
            return heuristic
        if not self._llm_available():
            return heuristic

        rule = rule_record.get("rule") if isinstance(rule_record, dict) and isinstance(rule_record.get("rule"), dict) else {}
        prompt = {
            "task": "Validate whether a physics diagnostic is publishable.",
            "labels": ["valid", "too_broad", "irrelevant", "duplicate", "needs_rewrite"],
            "question": (question or "")[:1200],
            "reference_answer": (reference_answer or "")[:1200],
            "student_solution": (student_solution or "")[:2500],
            "diagnostic": {
                "rule_id": diagnostic.get("rule"),
                "message": _message_text(diagnostic),
                "quote": _quote_text(diagnostic),
                "rule_title": rule.get("title") or rule.get("name"),
            },
            "requirements": [
                "Must identify a concrete physics error, not a vague warning.",
                "Must be supported by the exact quote.",
                "Must explain the correct physical relation when possible.",
                "Reject adjacent-topic triggers and final-answer-only critiques.",
            ],
            "output_schema": {"verdict": "string", "reasons": ["string"], "rewritten_message": "string|null"},
        }
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.llm_model,
                temperature=self.temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return JSON only. No thinking tags."},
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
            )
            raw = str(resp.choices[0].message.content or "").strip()
            parsed = json.loads(raw)
            verdict = str(parsed.get("verdict") or heuristic.get("verdict") or "valid").strip().lower()
            if verdict not in {"valid", "too_broad", "irrelevant", "duplicate", "needs_rewrite"}:
                verdict = str(heuristic.get("verdict") or "valid")
            return {
                "verdict": verdict,
                "reasons": list(parsed.get("reasons") or heuristic.get("reasons") or []),
                "rewritten_message": parsed.get("rewritten_message") or heuristic.get("rewritten_message"),
                "llm_validated": True,
            }
        except Exception as exc:  # pragma: no cover - runtime dependent
            out = dict(heuristic)
            out["llm_error"] = str(exc)
            return out

    def validate_one(
        self,
        *,
        question: str,
        reference_answer: str,
        student_solution: str,
        diagnostic: Dict[str, Any],
        rule_record: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        heuristic = self._heuristic_validate(
            question=question,
            reference_answer=reference_answer,
            student_solution=student_solution,
            diagnostic=diagnostic,
            rule_record=rule_record,
        )
        result = self._llm_validate(
            question=question,
            reference_answer=reference_answer,
            student_solution=student_solution,
            diagnostic=diagnostic,
            rule_record=rule_record,
            heuristic=heuristic,
        )
        verdict = str(result.get("verdict") or "valid")
        enriched = dict(diagnostic)
        enriched["validator"] = result

        if verdict in {"too_broad", "irrelevant", "duplicate"}:
            return None, {
                "reason": f"diagnostic_validator_{verdict}",
                "validator": result,
                "original_diagnostic": diagnostic,
            }
        if verdict == "needs_rewrite" and result.get("rewritten_message"):
            enriched["message"] = str(result.get("rewritten_message"))
        return enriched, None

    def validate_batch(
        self,
        *,
        question: str,
        reference_answer: str,
        student_solution: str,
        diagnostics: List[Dict[str, Any]],
        rule_records: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        record_by_rule = {
            str((item.get("rule") or {}).get("id") or item.get("rule_id") or ""): item
            for item in rule_records or []
            if isinstance(item, dict)
        }
        kept: List[Dict[str, Any]] = []
        suppressed: List[Dict[str, Any]] = []
        seen_quote_rule: set[Tuple[str, str]] = set()

        for diagnostic in diagnostics or []:
            if not isinstance(diagnostic, dict):
                continue
            rid = str(diagnostic.get("rule") or "")
            quote_key = (_quote_text(diagnostic).casefold(), rid)
            if quote_key in seen_quote_rule and quote_key[0]:
                suppressed.append(
                    {
                        "reason": "diagnostic_validator_duplicate",
                        "rule_id": rid,
                        "original_diagnostic": diagnostic,
                    }
                )
                continue

            kept_one, suppressed_one = self.validate_one(
                question=question,
                reference_answer=reference_answer,
                student_solution=student_solution,
                diagnostic=diagnostic,
                rule_record=record_by_rule.get(rid),
            )
            if suppressed_one:
                suppressed.append(suppressed_one)
                continue
            if kept_one:
                kept.append(kept_one)
                if quote_key[0]:
                    seen_quote_rule.add(quote_key)
        return kept, suppressed
