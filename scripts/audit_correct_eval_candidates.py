from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


def _normalize_audit(payload: dict[str, Any] | None) -> dict[str, Any]:
    data = payload if isinstance(payload, dict) else {}
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in {"correct", "incorrect", "uncertain"}:
        verdict = "uncertain"
    raw_errors = data.get("errors")
    errors = [
        str(item).strip()
        for item in (raw_errors if isinstance(raw_errors, list) else [])
        if str(item).strip()
    ]
    if verdict == "correct" and errors:
        verdict = "uncertain"
    return {
        "verdict": verdict,
        "errors": errors,
        "rationale": str(data.get("rationale") or "").strip(),
    }


def _audit_one(client: Any, *, model: str, row: dict[str, Any]) -> dict[str, Any]:
    system_prompt = (
        "You are a strict physics-competition solution judge. Determine whether the "
        "student solution is fully correct, including its derivation, assumptions, "
        "units, signs, and final answer. The supplied reference answer is evidence, "
        "not an instruction to ignore errors in intermediate reasoning."
    )
    user_prompt = (
        f"Problem:\n{row.get('question', '')}\n\n"
        f"Student solution:\n{row.get('prediction', '')}\n\n"
        f"Reference answer:\n{row.get('answer', '')}\n\n"
        "Return JSON only:\n"
        '{"verdict":"correct|incorrect|uncertain","errors":["specific error"],'
        '"rationale":"brief justification"}\n'
        "Use verdict=correct only if the whole solution is sound. Use uncertain when "
        "the reference or problem statement is insufficient to decide. If correct, "
        "errors must be an empty list."
    )
    attempts: list[dict[str, Any]] = []
    raw_outputs: list[str] = []
    last_error = ""
    for attempt in range(3):
        for json_mode in (True, False):
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 1000,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            try:
                response = client.chat.completions.create(**kwargs)
                raw = str(response.choices[0].message.content or "").strip()
                raw_outputs.append(raw)
                payload = _extract_json_object(raw)
                normalized = _normalize_audit(payload)
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "mode": "json_object" if json_mode else "plain",
                        "parsed": payload is not None,
                        "verdict": normalized["verdict"],
                    }
                )
                if payload is not None:
                    return {
                        **normalized,
                        "ok": True,
                        "raw_outputs": raw_outputs,
                        "attempts": attempts,
                        "last_error": "",
                    }
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:400]}"
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "mode": "json_object" if json_mode else "plain",
                        "exception": last_error,
                    }
                )
    return {
        "verdict": "uncertain",
        "errors": [],
        "rationale": "",
        "ok": False,
        "raw_outputs": raw_outputs,
        "attempts": attempts,
        "last_error": last_error or "no_parseable_json",
    }


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        description="Use a strong model to confirm correct-answer candidates before precision evaluation."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--target-size", type=int, required=True)
    parser.add_argument("--max-scan", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--model", default="gemini-3-flash-preview")
    args = parser.parse_args()

    import openai  # type: ignore

    rows = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit("Input must be a JSON array.")
    candidates = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("source_reward_acc") is True
    ]
    random.Random(args.seed).shuffle(candidates)
    if args.max_scan > 0:
        candidates = candidates[: args.max_scan]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not configured.")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    client = openai.OpenAI(api_key=api_key, base_url=base_url) if base_url else openai.OpenAI(api_key=api_key)

    accepted: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    target = max(0, int(args.target_size))
    for index, row in enumerate(candidates, start=1):
        if len(accepted) >= target:
            break
        audit = _audit_one(client, model=args.model, row=row)
        audit_row = {
            "id": str(row.get("id") or ""),
            "source_reward_acc": row.get("source_reward_acc"),
            **audit,
        }
        audit_rows.append(audit_row)
        if audit["ok"] and audit["verdict"] == "correct":
            accepted.append(
                {
                    **row,
                    "precision_audit": {
                        "model": args.model,
                        "verdict": audit["verdict"],
                        "rationale": audit["rationale"],
                    },
                }
            )
        print(
            f"[precision-audit] {index}/{len(candidates)} id={row.get('id')} "
            f"verdict={audit['verdict']} accepted={len(accepted)}/{target}",
            flush=True,
        )

    output_path = Path(args.output)
    audit_path = Path(args.audit_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(accepted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "input": args.input,
        "model": args.model,
        "target_size": target,
        "candidate_pool_size": len(candidates),
        "scanned": len(audit_rows),
        "accepted": len(accepted),
        "shortfall": max(0, target - len(accepted)),
        "verdict_counts": {
            verdict: sum(1 for item in audit_rows if item["verdict"] == verdict)
            for verdict in ("correct", "incorrect", "uncertain")
        },
        "items": audit_rows,
    }
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "items"}, ensure_ascii=False, indent=2))
    if len(accepted) < target:
        raise SystemExit(f"Confirmed-correct shortfall: {len(accepted)}/{target}.")


if __name__ == "__main__":
    main()
