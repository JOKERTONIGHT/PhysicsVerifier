#!/usr/bin/env python3
"""Build a human-edited SFT jsonl from the auto-repaired file.

Drops questions that cannot be solved from the given text (missing figure /
gold-fitting). Rewrites a few complete problems by hand. Applies light
editorial cleanup to the rest (emoji, translated-text headers).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.rl_data.screen_training_data import gold_fit_drop_reason, prompt_drop_reason
# Prefer the auto-repaired 398-row dump so re-running stays reproducible.
SRC = ROOT / "data/rl/sft_solutions.autorepair.jsonl"
if not SRC.is_file():
    SRC = ROOT / "data/rl/sft_solutions.jsonl"
DST = ROOT / "data/rl/sft_solutions.manual.jsonl"
REPORT = ROOT / "data/rl/sft_manual_edit_report.json"

# Incomplete stems: the visible text does not determine the gold answer.
DROP_IDS = {
    "154795",  # "flux through the loop" — geometry missing
    "162_917",  # process B — cycle missing
    "181_506",  # capillary heat — apparatus missing
    "228_537",  # ice thickness rate — numbers missing
    "244_792",  # nuclear X1..X7 — reactions missing
    "244_911",  # constants table + fragment
    "247_840",  # monopole N — setup missing
    "268_692",  # Earth-to-Mars launch — one-line stem
    "99596",  # T-t graph not shown
    "220_385",  # scale graph not shown
    "264_81",  # binary masses — orbital data missing
    "219_87",  # voltmeter A-B — circuit missing
    # Second pass: incomplete logic / gold-paste (spot check 2026-09-05)
    "83_148",  # stem has no circuit, Q0 invented
    "154_313",  # two problems concatenated; tensegrity figure missing
    "246_1",  # rod figure missing; h=L/2 inconsistent with point mass
    "84_794",  # invents T0, RH0 to hit 1310 m
    "222_364",  # circuit diagram missing; copies Is/9
    "156_1019",  # honest energy gives 2mgl/π, then pastes gold
    "257_426",  # crane diagram missing; reverse-engineers 10 kN
    "125_959",  # invents 500 nm from "standard glass"
    "256_947",  # calculation 125 m/s, boxes 50 because gold
    "251_322",  # sets p_B=220 hPa from gold then backsolves
    "91_1016",  # digit-hunting to match 282
    "232_110",  # Δn=0.013 not in the stem
    "133_545",  # derivation ≠ gold; "accept reference result"
    "230_931",  # counts 1–3 states then forces gold=1
    "82_624",  # geometry conflict; follows gold formula
    "110_952",  # 946 because gold, not from a closed count
    "113_88",  # changes order of splitting to match T_beats
    "138_259",  # four forces asked; boxes 0,0,0,0 with missing figure
    "196_995",  # invents ATLAS inner radius 1.15 m
    "189_545",  # invents window area to hit 21.47 yuan
    "146_700",  # skips the integral, pastes gold B0
    "100_135",  # radial-force sign/factor unresolved, pastes gold
    "268_152",  # off by 4× then "thus the given boxed result"
    "108_114",  # invents 0.1 Hz heart rate to hit 8e-5
    "132476",  # two unrelated numbered problems glued
}

EMOJI_RE = re.compile(r"[✅📌]")
HEADER_RE = re.compile(r"(?im)^here's the translated text:?\s*")
NOTE_LINE_RE = re.compile(r"(?m)^\s*>\s*📌.*$")


def assistant_text(row: dict) -> str:
    for msg in reversed(row.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return str(msg.get("content") or "")
    return ""


def user_text(row: dict) -> str:
    for msg in row.get("messages") or []:
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content") or "")
    return str(row.get("question") or "")


def set_assistant(row: dict, text: str) -> dict:
    out = dict(row)
    messages = [dict(m) for m in (row.get("messages") or [])]
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            messages[i]["content"] = text
            break
    else:
        messages.append({"role": "assistant", "content": text})
    out["messages"] = messages
    return out


def set_user(row: dict, text: str) -> dict:
    out = dict(row)
    messages = [dict(m) for m in (row.get("messages") or [])]
    for i, msg in enumerate(messages):
        if msg.get("role") == "user":
            messages[i]["content"] = text
            break
    out["messages"] = messages
    if "question" in out:
        out["question"] = text
    return out


# OCR answer keys that leaked into the stored user turn.
USER_FIX = {
    "83_420": lambda u: re.sub(
        r"\n+Figure 4\s*\n+\(9\)\s*\$P \+ \\frac\{mg\}\{\\pi r\^2\}\$\s*$",
        "",
        u,
    ).rstrip(),
}


def editorial_clean(text: str) -> str:
    text = EMOJI_RE.sub("", text)
    text = NOTE_LINE_RE.sub("", text)
    text = HEADER_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


# Hand-written replacements: only for stems that actually contain the physics.
MANUAL = {
    "83_420": r"""The inverted tube (open end down) traps an air column of length \(l_i\). The gas above the free liquid surface has pressure \(P\). The glass volume is negligible, so buoyancy on the glass itself can be ignored, and surface tension is neglected.

Consider vertical forces on the tube. The sealed end has area \(A=\pi r^2\). Outside it, the container gas pushes down with force \(PA\). Inside, the trapped air pushes up with force \(P_{\mathrm{in}}A\). The tube weight \(mg\) acts down. In equilibrium
\[
P_{\mathrm{in}}A = PA + mg,
\]
hence
\[
P_{\mathrm{in}} = P + \frac{mg}{\pi r^2}.
\]
This is the pressure of the air column inside the tube (part 9).

\boxed{P+\frac{mg}{\pi r^2}}
""",
    "6957": r"""Both spheres fall freely from rest through height \(H\), so just before the heavy sphere hits the ground they have the same downward speed \(u=\sqrt{2gH}\).

The ground collision is perfectly elastic, so the heavy sphere's velocity reverses to \(+u\) (upward) while the light sphere, which has not yet hit the ground, still has velocity \(-u\).

The subsequent sphere–sphere collision is one-dimensional and perfectly elastic. With incoming velocities \(v_M=+u\) and \(v_m=-u\),
\[
v_m' = \frac{(m-M)(-u)+2Mu}{M+m} = \frac{(3M-m)u}{M+m}.
\]
Immediately after this collision the light sphere's centre is at height \(3r\). It then rises an extra height \(v_m'^2/(2g)\). The measured maximum height of its centre is \(3r+4H\), so
\[
\frac{v_m'^2}{2g}=4H \qquad\Rightarrow\qquad v_m'=2u.
\]
Therefore \((3M-m)/(M+m)=2\), which gives \(M/m=3\).

\boxed{3}
""",
    "175_720": r"""Until the first hit on the sheet, the electric field \(\mathbf{E}=-E\hat{\mathbf{z}}\) governs the \(z\)-motion and the magnetic field (parallel to \(z\)) does not change the speed in the \(xy\) plane.

Write \(v_z=v\sin\theta\) and \(v_\perp=v\cos\theta\). The downward acceleration is \(a=qE/m\), so the particle returns to \(z=0\) after
\[
t=\frac{2v\sin\theta}{a}=\frac{2mv\sin\theta}{qE}.
\]
The path length of the *projection* onto \(xOy\) is the distance travelled at constant speed \(v_\perp\),
\[
s=v_\perp t=\frac{2mv^2}{qE}\sin\theta\cos\theta=\frac{mv^2}{qE}\sin 2\theta.
\]
This is largest at \(\theta=\pi/4\):
\[
s_{\max}=\frac{mv^2}{qE}.
\]

\boxed{\frac{m v^{2}}{q E}}
""",
}


def main() -> None:
    rows = [json.loads(x) for x in SRC.read_text(encoding="utf-8").splitlines() if x.strip()]
    kept = []
    dropped = []
    rewritten = []
    cleaned = 0
    for row in rows:
        sid = str(row.get("sample_id") or "")
        if sid in DROP_IDS:
            dropped.append(sid)
            continue
        if prompt_drop_reason(row):
            dropped.append(sid)
            continue
        if sid in MANUAL:
            rec = set_assistant(row, MANUAL[sid].strip())
            if sid in USER_FIX:
                rec = set_user(rec, USER_FIX[sid](user_text(rec)))
            rec["manual_edit"] = "rewrite"
            rewritten.append(sid)
            kept.append(rec)
            continue
        asst = assistant_text(row)
        new = editorial_clean(asst)
        rec = dict(row)
        if new != asst:
            rec = set_assistant(rec, new)
            rec["manual_edit"] = "editorial"
            cleaned += 1
        if gold_fit_drop_reason(assistant_text(rec)):
            dropped.append(sid)
            continue
        kept.append(rec)

    # Add the hand-fixed rejected row whose stem is complete.
    have = {str(r.get("sample_id") or "") for r in kept}
    rej_path = ROOT / "data/rl/sft_solutions_rejected.jsonl"
    if rej_path.is_file():
        for line in rej_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sid = str(row.get("sample_id") or "")
            if sid == "83_420" and sid in MANUAL and sid not in have:
                if prompt_drop_reason(row):
                    continue
                rec = set_assistant(row, MANUAL[sid].strip())
                rec = set_user(rec, USER_FIX[sid](user_text(rec)))
                rec.pop("audit_flags", None)
                rec["manual_edit"] = "rewrite"
                rewritten.append(sid)
                kept.append(rec)
                have.add(sid)
    DST.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
        encoding="utf-8",
    )
    report = {
        "n_in": len(rows),
        "n_out": len(kept),
        "n_dropped_incomplete": len(dropped),
        "dropped_ids": sorted(dropped),
        "n_rewritten": len(rewritten),
        "rewritten_ids": rewritten,
        "n_editorial_clean": cleaned,
        "output": str(DST),
    }
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
