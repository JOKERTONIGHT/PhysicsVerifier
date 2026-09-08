#!/usr/bin/env python3
"""SFT-side answer equivalence: units, SI prefixes, and numeric tolerance.

Does not change ``grade_answer_verl``. Reward/eval scoring stays strict; this
helper is only for SFT rejection sampling, where gold often carries ``\\mathrm{~N}``
or a leading ``X=``.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

from training.compat.math_grading import extract_answer, grade_answer_verl

REL_TOL = 5e-3

LATEX_UNIT_RE = re.compile(
    r"\\(?:mathrm|text|mathbf|operatorname|textrm)\s*\{[^{}]*\}"
)
LATEX_SPACE_RE = re.compile(r"\\[,;:!~\s]")
LHS_RE = re.compile(
    r"^(?:\\(?:mathrm|text|mathbf)\{)?[A-Za-z][A-Za-z0-9_\\^{}]{0,24}(?:\})?\s*(?:\\approx|≈|=)\s*"
)
APPROX_RE = re.compile(r"\\approx|≈")
SCI_RE = re.compile(
    r"(-?\d+(?:\.\d+)?)\s*(?:\\(?:times|cdot)\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?|[eE]([+-]?\d+))"
)
PLAIN_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
FRAC_RE = re.compile(r"\\(?:frac|dfrac|tfrac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}")

# Longest SI unit tokens first so "MeV" / "μm" win over "m".
SI_UNITS = (
    "ohm",
    "mol",
    "rad",
    "deg",
    "MeV",
    "GeV",
    "keV",
    "eV",
    "Pa",
    "Hz",
    "kg",
    "km",
    "cm",
    "mm",
    "nm",
    "μm",
    "um",
    "µs",
    "ms",
    "ns",
    "μs",
    "mus",
    "kN",
    "mN",
    "kJ",
    "mJ",
    "kW",
    "mW",
    "kV",
    "mV",
    "mA",
    "kA",
    "m",
    "s",
    "N",
    "J",
    "W",
    "K",
    "V",
    "A",
    "C",
    "T",
    "g",
    "H",
    "F",
    "Ω",
    "°",
)
SI_UNIT_RE = re.compile(r"(?:" + "|".join(re.escape(u) for u in SI_UNITS) + r")(?![A-Za-z])")
PREFIX_SCALE = {
    "p": 1e-12,
    "n": 1e-9,
    "μ": 1e-6,
    "µ": 1e-6,
    "u": 1e-6,
    "m": 1e-3,
    "c": 1e-2,
    "k": 1e3,
    "M": 1e6,
    "G": 1e9,
    "T": 1e12,
}
BASE_DIM = {
    "m": "m",
    "s": "s",
    "g": "kg",
    "kg": "kg",
    "n": "N",
    "j": "J",
    "w": "W",
    "pa": "Pa",
    "hz": "Hz",
    "k": "K",
    "ev": "eV",
    "v": "V",
    "a": "A",
    "c": "C",
    "t": "T",
    "ohm": "ohm",
    "ω": "ohm",
    "Ω": "ohm",
    "mol": "mol",
    "rad": "rad",
    "deg": "rad",
    "°": "rad",
    "h": "H",
    "f": "F",
}


def _payload(text: str) -> str:
    src = str(text or "").strip()
    if "\\boxed" in src:
        extracted = extract_answer(src)
        if extracted:
            return extracted.strip()
    return src.strip().strip("$").strip()


def strip_units(text: str) -> str:
    src = str(text or "")
    src = src.replace("~", " ").replace("\\,", " ").replace("\\;", " ")
    src = LATEX_UNIT_RE.sub(" ", src)
    src = LATEX_SPACE_RE.sub(" ", src)
    src = src.replace("\\mathrm", " ").replace("\\text", " ")
    src = src.replace("μ", "μ").replace("µ", "μ")
    src = SI_UNIT_RE.sub(" ", src)
    src = src.replace("°", " ").replace("Ω", " ")
    src = APPROX_RE.sub(" ", src)
    src = LHS_RE.sub("", src.strip())
    src = re.sub(r"[{}]", "", src)
    src = re.sub(r"\s+", "", src)
    return src


def _parse_sci(text: str) -> Optional[float]:
    src = str(text or "")
    src = FRAC_RE.sub(
        lambda m: str(float(m.group(1)) / float(m.group(2)))
        if _is_plain_number(m.group(1)) and _is_plain_number(m.group(2))
        else m.group(0),
        src,
    )
    m = SCI_RE.search(src)
    if m:
        base = float(m.group(1))
        exp = m.group(2) or m.group(3)
        return base * (10 ** int(exp))
    m = PLAIN_NUM_RE.search(src.replace("{", "").replace("}", ""))
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def _is_plain_number(text: str) -> bool:
    try:
        float(text)
        return True
    except (TypeError, ValueError):
        return False


def _split_number_unit(text: str) -> tuple[Optional[float], str]:
    src = _payload(text)
    src = src.replace("~", " ")
    src = LATEX_UNIT_RE.sub(lambda m: " " + m.group(0).split("{", 1)[-1].rstrip("}"), src)
    src = LATEX_SPACE_RE.sub(" ", src)
    src = APPROX_RE.sub(" ", src)
    src = LHS_RE.sub("", src.strip())
    value = _parse_sci(src)
    unit_src = re.sub(r"\\(?:times|cdot|mathrm|text|mathbf|operatorname)\s*", " ", src)
    unit_src = re.sub(r"[{}\\]", " ", unit_src)
    unit_m = SI_UNIT_RE.search(unit_src)
    unit = unit_m.group(0) if unit_m else ""
    return value, unit


def _to_si(value: float, unit: str) -> tuple[float, str]:
    raw = (unit or "").strip()
    if not raw:
        return value, ""
    if raw.lower() in {"deg", "°"}:
        return value, "rad"
    if raw in {"μm", "um"}:
        return value * 1e-6, "m"
    if len(raw) >= 2 and raw[0] in PREFIX_SCALE:
        prefix = raw[0]
        rest = raw[1:]
        dim = BASE_DIM.get(rest.lower(), rest.lower())
        scale = PREFIX_SCALE[prefix]
        if rest.lower() == "g" and prefix.lower() != "k":
            scale *= 1e-3
            dim = "kg"
        return value * scale, dim
    dim = BASE_DIM.get(raw.lower(), raw.lower())
    if raw.lower() == "g":
        return value * 1e-3, "kg"
    return value, dim


def answers_equivalent(cand: str, gold: str, *, rel_tol: float = REL_TOL) -> Tuple[bool, str]:
    """Return (ok, reason) for SFT matching. Never used as the RL reward."""
    if not gold or not cand:
        return False, "-"
    if grade_answer_verl(cand, gold):
        return True, "verl"
    given = _payload(cand)
    truth = _payload(gold)
    if not given or not truth:
        return False, "-"
    g_stripped = strip_units(given)
    t_stripped = strip_units(truth)
    if g_stripped and g_stripped == t_stripped:
        return True, "unit_strip"
    gv, gu = _split_number_unit(given)
    tv, tu = _split_number_unit(truth)
    if gv is not None and tv is not None:
        g_si, g_dim = _to_si(gv, gu)
        t_si, t_dim = _to_si(tv, tu)
        dims_ok = (not g_dim or not t_dim) or g_dim == t_dim
        denom = abs(t_si) if t_si != 0 else abs(g_si)
        if dims_ok and denom != 0 and abs(g_si - t_si) / denom <= rel_tol:
            return True, "numeric"
        if dims_ok and denom == 0 and abs(g_si - t_si) <= rel_tol:
            return True, "numeric"
    boxed_g = given if given.startswith("\\boxed") else f"\\boxed{{{g_stripped or given}}}"
    if grade_answer_verl(boxed_g, t_stripped or truth):
        return True, "verl_stripped"
    if g_stripped and t_stripped and grade_answer_verl(f"\\boxed{{{g_stripped}}}", t_stripped):
        return True, "verl_stripped"
    return False, "-"
