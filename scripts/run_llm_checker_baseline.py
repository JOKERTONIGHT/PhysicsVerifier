import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import openai
from tqdm import tqdm
from dotenv import load_dotenv

def extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    def _latex_friendly_json(s: str) -> str:
        # LLMs often emit LaTeX ``\( ... \)`` inside JSON strings; ``\(`` is not a valid JSON escape.
        return (
            s.replace("\\(", "(")
            .replace("\\)", ")")
            .replace("\\[", "[")
            .replace("\\]", "]")
        )

    last_err: Optional[Exception] = None
    for candidate in (text, _latex_friendly_json(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as e:
            last_err = e
    print(f"Failed to parse JSON: {last_err}\nRaw text: {text[:2000]}")
    return {"diagnostics": []}


def _openai_disable_thinking_kwargs() -> Dict[str, Any]:
    flag = str(os.getenv("OPENAI_DISABLE_THINKING", "")).strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


def normalize_baseline_diagnostics(parsed: Any) -> List[Dict[str, Any]]:
    """Semantic baseline outputs message + evidence.quote only (no rule labels) for fair evaluation."""
    root: Dict[str, Any] = parsed if isinstance(parsed, dict) else {}
    raw = root.get("diagnostics")
    raw_list = raw if isinstance(raw, list) else []
    out: List[Dict[str, Any]] = []
    for d in raw_list:
        if not isinstance(d, dict):
            continue
        ev_raw = d.get("evidence")
        ev = ev_raw if isinstance(ev_raw, dict) else {}
        quote = str(ev.get("quote") or "").strip()
        msg = str(d.get("message") or "").strip()
        if not msg and not quote:
            continue
        item: Dict[str, Any] = {"message": msg, "evidence": {"quote": quote}}
        out.append(item)
    return out


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input dataset JSON path")
    parser.add_argument("--model", type=str, default="qwen3-30b-a3b-instruct-2507", help="Model name")
    parser.add_argument("--out_json", type=str, required=True, help="Output JSON path")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Write partial results to disk every N samples (default 1) for crash-safe monitoring.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request HTTP timeout in seconds (slow gateways may need 120+).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=10,
        metavar="N",
        help="Print a throughput summary every N completed samples; 0 disables.",
    )
    parser.add_argument(
        "--no-tqdm",
        action="store_true",
        help="Disable tqdm progress bar (cleaner logs under screen/tmux).",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(
        f"[baseline] loaded {len(data)} samples from {args.input!r} | "
        f"model={args.model!r} | progress every {max(0, int(args.progress_interval))} (0=off) | "
        f"no_tqdm={bool(args.no_tqdm)}",
        flush=True,
    )

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE")
    api_key = os.getenv("OPENAI_API_KEY", "dummy_key")
    
    if not base_url:
        print("WARNING: OPENAI_BASE_URL not set. Using default openai endpoint.")
        client = openai.OpenAI(api_key=api_key)
    else:
        print(f"Using base_url: {base_url}")
        client = openai.OpenAI(base_url=base_url, api_key=api_key)

    results: List[Dict[str, Any]] = []
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    flush_every = max(1, int(args.flush_every))
    progress_interval = max(0, int(args.progress_interval))
    total = len(data)
    t0 = time.perf_counter()

    def _fmt_duration(sec: float) -> str:
        if sec < 60.0:
            return f"{sec:.1f}s"
        sec_i = int(sec)
        m, s = divmod(sec_i, 60)
        if m < 60:
            return f"{m}m{s}s"
        h, m = divmod(m, 60)
        return f"{h}h{m}m{s}s"

    def _progress_line(done: int, last_id: Any) -> None:
        if progress_interval <= 0:
            return
        if done % progress_interval != 0 and done != total:
            return
        elapsed = time.perf_counter() - t0
        rate = elapsed / done if done else 0.0
        print(
            f"[baseline] progress {done}/{total} samples | "
            f"elapsed {_fmt_duration(elapsed)} | "
            f"avg {rate:.2f}s/sample | last_id={last_id!r}",
            flush=True,
        )

    def _persist() -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    system_prompt = """You are an expert physics evaluator.
Your task is to review a student's step-by-step solution to a physics problem and identify ANY physics errors, logic errors, or formula misapplications.
Do NOT flag minor rounding differences as errors. Only flag genuine physical or mathematical mistakes.
Do NOT name or cite catalog rules or rule IDs; describe the mistake in plain language only.
If you find errors, you MUST extract the exact quote from the student's solution where the error occurs (for localization and evaluation).

Output your findings STRICTLY as a JSON object with the following schema:
{
  "diagnostics": [
    {
      "message": "Clear explanation of what is wrong and why it violates physics or reasoning.",
      "evidence": {
        "quote": "The EXACT matching substring from the student's solution that contains the error. Do not truncate or modify it."
      }
    }
  ]
}
If the solution is completely correct and has no errors, output {"diagnostics": []}.
Return ONLY the JSON object without any markdown formatting or extra text."""

    if args.no_tqdm:
        sample_iter = data
    else:
        sample_iter = tqdm(data, desc="Evaluating with LLM Baseline", total=total)

    for item in sample_iter:
        item_id = item.get("id")
        question = item.get("question", "")
        prediction = item.get("prediction", "")
        
        user_msg = f"--- Problem ---\n{question}\n\n--- Student Solution ---\n{prediction}\n\nReview the solution and output the JSON diagnostics."
        
        parsed = {"diagnostics": []}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=args.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.1,
                    max_tokens=2048,
                    timeout=float(args.timeout),
                    **_openai_disable_thinking_kwargs(),
                )
                content = resp.choices[0].message.content
                parsed = extract_json(content)
                break
            except Exception as e:
                print(f"Error evaluating sample {item_id} (Attempt {attempt+1}/{max_retries}): {e}")
                time.sleep(5)
            
        res_item = {
            "id": item_id,
            "topic": item.get("topic", "Unknown"),
            "verifier": f"baseline_llm_{args.model}",
            "diagnostics": normalize_baseline_diagnostics(parsed),
        }
        results.append(res_item)
        if len(results) % flush_every == 0:
            _persist()
        _progress_line(len(results), item_id)

    _persist()
    wall = time.perf_counter() - t0
    print(f"Saved {len(results)} results to {out_path} (wall {_fmt_duration(wall)})", flush=True)

if __name__ == "__main__":
    main()
