"""Stream-parse top-level JSON array of rollout batches from very large files.

The Physics rollout dump (`data/combined_language_only.json`) is a JSON array:
`[ { \"rollout_id\": ..., \"samples\": [ ... ] }, ... ]`

Standard `json.load` would require RAM proportional to file size; this module
yields one rollout dict at a time using `JSONDecoder.raw_decode`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, TextIO


def _consume_until_array_start(fp: TextIO) -> None:
    while True:
        ch = fp.read(1)
        if not ch:
            raise ValueError("EOF before JSON array start '['")
        if ch.isspace():
            continue
        if ch == "[":
            return
        raise ValueError(f"Expected '[' start of array, got {ch!r}")


def iter_rollout_batches(path: Path | str, *, chunk_bytes: int = 8 * 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Yield each element of the outer JSON array (one rollout batch)."""
    decoder = json.JSONDecoder()
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="replace") as fp:
        _consume_until_array_start(fp)
        buf = ""
        while True:
            buf = buf.lstrip()
            if buf.startswith("]"):
                break
            if buf.startswith(","):
                buf = buf[1:].lstrip()

            while True:
                if not buf:
                    more = fp.read(chunk_bytes)
                    if not more:
                        raise ValueError("Truncated JSON array (unexpected EOF)")
                    buf += more
                    continue
                try:
                    buf_stripped = buf.lstrip()
                    obj, idx = decoder.raw_decode(buf_stripped)
                    buf = buf_stripped[idx:]
                    if isinstance(obj, dict):
                        yield obj
                    break
                except json.JSONDecodeError:
                    more = fp.read(chunk_bytes)
                    if not more:
                        buf = buf.lstrip()
                        if buf.startswith("]") or not buf:
                            break
                        raise ValueError("Incomplete JSON element before EOF (truncated file?)") from None
                    buf += more


def count_rollouts_and_samples(path: Path | str, *, max_rollouts: Optional[int] = None) -> Dict[str, Any]:
    """Lightweight stats without loading full structures into RAM."""
    rollouts = 0
    samples = 0
    for batch in iter_rollout_batches(path):
        rollouts += 1
        s = batch.get("samples")
        if isinstance(s, list):
            samples += len(s)
        if max_rollouts is not None and rollouts >= max_rollouts:
            break
    return {"rollouts_seen": rollouts, "samples_seen": samples}
