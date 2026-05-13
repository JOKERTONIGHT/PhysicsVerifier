#!/usr/bin/env python3
"""Stream-audit `data/combined_language_only.json` (large rollout dump).

Produces JSON + Markdown summaries for data quality review before building eval slices.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.combined_language_io import iter_rollout_batches  # noqa: E402
from scripts.combined_language_samples import parse_reward_field, extract_question  # noqa: E402


def _audit_rollout_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    samples = batch.get("samples") if isinstance(batch.get("samples"), list) else []
    lens: List[int] = []
    acc_flags: List[bool] = []
    empty_q = 0
    empty_resp = 0
    for s in samples:
        if not isinstance(s, dict):
            continue
        rlen = int(s.get("response_length") or len(str(s.get("response") or "")))
        lens.append(rlen)
        rw = parse_reward_field(s.get("reward"))
        if "acc" in rw:
            acc_flags.append(bool(rw.get("acc")))
        q = extract_question(s)
        if not str(q).strip():
            empty_q += 1
        if not str(s.get("response") or "").strip():
            empty_resp += 1

    return {
        "rollout_id": batch.get("rollout_id"),
        "sample_count": len(samples),
        "empty_question_count": empty_q,
        "empty_response_count": empty_resp,
        "response_length_mean": round(statistics.mean(lens), 2) if lens else 0.0,
        "response_length_p95": round(_pctl(lens, 95), 2) if lens else 0.0,
        "reward_acc_available_count": len(acc_flags),
        "reward_acc_true_ratio": round(sum(1 for x in acc_flags if x) / len(acc_flags), 4) if acc_flags else None,
    }


def _pctl(values: List[int], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
    return float(xs[k])


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit combined_language_only rollout JSON (streaming).")
    parser.add_argument(
        "--input",
        type=str,
        default=str(REPO_ROOT / "data" / "combined_language_only.json"),
    )
    parser.add_argument("--max-rollouts", type=int, default=0, help="0 = scan all rollout batches (slow on multi-GB files).")
    parser.add_argument("--out-json", type=str, default=str(REPO_ROOT / "data" / "derived" / "combined_language_audit.json"))
    parser.add_argument("--out-md", type=str, default=str(REPO_ROOT / "docs" / "combined_language_audit_report.md"))
    args = parser.parse_args()

    src = Path(args.input)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    rollouts = 0
    samples_total = 0
    per_rollout: List[Dict[str, Any]] = []
    max_r = int(args.max_rollouts or 0)

    for batch in iter_rollout_batches(src):
        rollouts += 1
        st = _audit_rollout_batch(batch)
        per_rollout.append(st)
        samples_total += int(st.get("sample_count") or 0)
        if max_r > 0 and rollouts >= max_r:
            break

    summary = {
        "source": str(src),
        "rollouts_audited": rollouts,
        "samples_total": samples_total,
        "max_rollouts_cap": max_r or None,
        "per_rollout": per_rollout,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Combined language rollout audit",
        "",
        f"- source: `{src}`",
        f"- rollout batches scanned: **{rollouts}**",
        f"- samples counted: **{samples_total}**",
        f"- cap: `{max_r or 'none (full pass)'}`",
        "",
        "Per-batch statistics are stored in JSON (`per_rollout`).",
        "",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {out_md}")


if __name__ == "__main__":
    main()
