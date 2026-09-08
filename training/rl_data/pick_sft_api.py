#!/usr/bin/env python3
"""Pick a working remote OpenAI-compatible pair from .env.

SFT_API_PAIR=1 uses the first pair (the key on .env line 2). 0 = first usable.
SFT_API_MODEL, if set, is the smoke-test / returned model name.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from openai import OpenAI

PREFER = ["deepseek-v4-flash", "gemini-2.5-flash", "gpt-4o-mini"]


def load_pairs(env_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    base: str | None = None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("OPENAI_BASE_URL="):
            base = s.split("=", 1)[1].strip().strip('"').strip("'")
        elif s.startswith("OPENAI_API_KEY=") and base:
            key = s.split("=", 1)[1].strip().strip('"').strip("'")
            pairs.append((base, key))
            base = None
    return pairs


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    pair_idx = int(os.environ.get("SFT_API_PAIR") or (sys.argv[2] if len(sys.argv) > 2 else "0"))
    prefer_model = os.environ.get("SFT_API_MODEL") or ""
    env_path = root / ".env"
    pairs = load_pairs(env_path)
    indexed = list(enumerate(pairs, 1))
    if pair_idx > 0:
        if pair_idx > len(pairs):
            print(f"[sft-gen] SFT_API_PAIR={pair_idx} missing ({len(pairs)} pairs)", file=sys.stderr)
            return 2
        indexed = [indexed[pair_idx - 1]]
    for i, (base, key) in indexed:
        client = OpenAI(base_url=base.rstrip("/"), api_key=key)
        try:
            ids = [m.id for m in client.models.list().data]
            candidates: list[str] = []
            if prefer_model:
                candidates.append(prefer_model)
            for name in PREFER:
                if name not in candidates:
                    candidates.append(name)
            chosen = ""
            last_exc: Exception | None = None
            require_smoke = os.environ.get("SFT_REQUIRE_SMOKE", "0") == "1"
            for model in candidates:
                last_exc = None
                for attempt in range(3):
                    try:
                        client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "Reply with the single word pong"}],
                            max_tokens=8,
                            timeout=30,
                        )
                        last_exc = None
                        chosen = model
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        if type(exc).__name__ not in {
                            "InternalServerError",
                            "APIConnectionError",
                            "RateLimitError",
                            "APITimeoutError",
                        }:
                            break
                        import time

                        time.sleep(2 ** attempt)
                if chosen:
                    model = chosen
                    break
                print(
                    f"[sft-gen] pair {i} {model} smoke failed ({type(last_exc).__name__ if last_exc else 'err'})",
                    file=sys.stderr,
                )
            if not chosen:
                if require_smoke:
                    continue
                model = prefer_model if prefer_model else next((n for n in PREFER if n in ids), PREFER[0])
                print(
                    f"[sft-gen] pair {i} smoke chat failed; models.list ok, using {model}",
                    file=sys.stderr,
                )
        except Exception as exc:  # noqa: BLE001
            print(f"[sft-gen] .env API pair {i} unusable: {type(exc).__name__}", file=sys.stderr)
            if pair_idx > 0:
                return 2
            continue
        # stdout: base, key, model, pair_index — consumed by launch_sft_datagen.sh
        print(base)
        print(key)
        print(model)
        print(i)
        return 0
    print("no usable OPENAI_* pair in .env", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
