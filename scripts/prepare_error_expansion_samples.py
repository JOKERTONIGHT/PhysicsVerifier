from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, Dict, Generator, List, Optional


def _iter_top_level_array(path: Path, chunk_size: int = 1 << 20) -> Generator[Any, None, None]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        pos = 0
        started = False
        ended = False
        eof = False

        while True:
            if not ended and not eof and (pos == len(buf) or (len(buf) - pos) < (chunk_size // 2)):
                chunk = f.read(chunk_size)
                if chunk:
                    if pos > 0:
                        buf = buf[pos:] + chunk
                        pos = 0
                    else:
                        buf += chunk
                else:
                    eof = True
                    if pos > 0:
                        buf = buf[pos:]
                        pos = 0

            need_more = False
            while True:
                while pos < len(buf) and buf[pos] in " \t\r\n":
                    pos += 1

                if not started:
                    if pos >= len(buf):
                        break
                    if buf[pos] != "[":
                        raise ValueError("Input JSON must be a top-level array.")
                    started = True
                    pos += 1
                    continue

                while pos < len(buf) and buf[pos] in " \t\r\n":
                    pos += 1

                if pos >= len(buf):
                    break

                ch = buf[pos]
                if ch == "]":
                    ended = True
                    pos += 1
                    break
                if ch == ",":
                    pos += 1
                    continue

                try:
                    obj, nxt = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    need_more = True
                    break

                yield obj
                pos = nxt

            if ended:
                return

            if need_more and not eof:
                chunk = f.read(chunk_size)
                if chunk:
                    if pos > 0:
                        buf = buf[pos:] + chunk
                        pos = 0
                    else:
                        buf += chunk
                    continue
                eof = True

            if eof:
                raise ValueError("Unexpected end of JSON while parsing array.")


def _safe_id(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _pick_question(sample: Dict[str, Any]) -> str:
    meta = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    question = str(meta.get("question") or "").strip()
    if question:
        return question

    prompt = str(sample.get("prompt") or "")
    marker = "Problem:"
    idx = prompt.rfind(marker)
    if idx >= 0:
        return prompt[idx + len(marker):].strip()
    return prompt.strip()


def _pick_prediction(sample: Dict[str, Any]) -> str:
    return str(sample.get("response") or sample.get("model_response") or "").strip()


def _pick_answer(sample: Dict[str, Any]) -> str:
    label = sample.get("label")
    if isinstance(label, list):
        if len(label) == 1:
            return str(label[0] or "").strip()
        return json.dumps(label, ensure_ascii=False)
    if label is not None:
        return str(label).strip()

    gt = sample.get("ground_truth_label")
    if isinstance(gt, list):
        if len(gt) == 1:
            return str(gt[0] or "").strip()
        return json.dumps(gt, ensure_ascii=False)
    if gt is not None:
        return str(gt).strip()

    return ""


def _is_wrong_sample(sample: Dict[str, Any]) -> bool:
    reward = sample.get("reward") if isinstance(sample.get("reward"), dict) else {}
    if "acc" in reward and isinstance(reward.get("acc"), bool):
        return reward.get("acc") is False

    for key in ("score", "score_noxverify", "point", "point_noxverify"):
        val = reward.get(key)
        if isinstance(val, (int, float)) and val < 0.999999:
            return True

    status = str(sample.get("status") or "").strip().lower()
    if status and status != "completed":
        return True

    return False


def _to_eval_item(sample: Dict[str, Any], rid: Any, sample_idx: int) -> Dict[str, Any]:
    sid = _safe_id(sample.get("id"), fallback=f"{rid}_{sample_idx}")
    return {
        "id": sid,
        "question": _pick_question(sample),
        "prediction": _pick_prediction(sample),
        "answer": _pick_answer(sample),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract wrong-answer samples from combined_language_only.json into evaluation format.",
    )
    parser.add_argument("--input", type=str, default="data/combined_language_only.json")
    parser.add_argument("--output", type=str, default="data/evaluation_sample_1000_expansion.json")
    parser.add_argument("--target-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260331)
    parser.add_argument("--chunk-size", type=int, default=1 << 20)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Print progress every N processed samples (set 0 to disable).",
    )
    parser.add_argument(
        "--allow-nonwrong-fallback",
        action="store_true",
        help="If wrong samples are insufficient, allow adding non-wrong samples to reach target-size.",
    )
    args = parser.parse_args()

    if args.target_size <= 0:
        raise SystemExit("--target-size must be > 0")

    in_path = Path(args.input)
    out_path = Path(args.output)
    rng = random.Random(args.seed)

    reservoir: List[Dict[str, Any]] = []
    wrong_seen = 0
    total_seen = 0

    fallback_pool: List[Dict[str, Any]] = []
    fallback_seen = 0

    for rollout in _iter_top_level_array(in_path, chunk_size=args.chunk_size):
        if not isinstance(rollout, dict):
            continue
        rid = rollout.get("rollout_id") or rollout.get("rollout") or "rollout"
        samples = rollout.get("samples") or []
        if not isinstance(samples, list):
            continue

        for i, sample in enumerate(samples):
            total_seen += 1
            if not isinstance(sample, dict):
                continue

            if _is_wrong_sample(sample):
                wrong_seen += 1
                if len(reservoir) < args.target_size:
                    item = _to_eval_item(sample, rid, i)
                    if item["question"] and item["prediction"]:
                        reservoir.append(item)
                else:
                    j = rng.randint(0, wrong_seen - 1)
                    if j < args.target_size:
                        item = _to_eval_item(sample, rid, i)
                        if item["question"] and item["prediction"]:
                            reservoir[j] = item
            elif args.allow_nonwrong_fallback:
                fallback_seen += 1
                if len(fallback_pool) < args.target_size:
                    item = _to_eval_item(sample, rid, i)
                    if item["question"] and item["prediction"]:
                        fallback_pool.append(item)
                else:
                    j = rng.randint(0, fallback_seen - 1)
                    if j < args.target_size:
                        item = _to_eval_item(sample, rid, i)
                        if item["question"] and item["prediction"]:
                            fallback_pool[j] = item

            if args.progress_every > 0 and total_seen % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "progress_total_seen": total_seen,
                            "progress_wrong_seen": wrong_seen,
                            "reservoir_size": len(reservoir),
                        },
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )

    out_items = list(reservoir)
    if len(out_items) < args.target_size and args.allow_nonwrong_fallback:
        needed = args.target_size - len(out_items)
        out_items.extend(fallback_pool[:needed])

    out_items.sort(key=lambda x: str(x.get("id") or ""))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_items, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "input": str(in_path),
        "output": str(out_path),
        "total_seen": total_seen,
        "wrong_seen": wrong_seen,
        "target_size": args.target_size,
        "actual_size": len(out_items),
        "allow_nonwrong_fallback": bool(args.allow_nonwrong_fallback),
        "seed": args.seed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
