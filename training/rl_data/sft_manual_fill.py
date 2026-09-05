#!/usr/bin/env python3
"""Append a batch of hand-written SFT solutions for complete, text-only stems.

Requires data/rl/swift_prompts.jsonl (gitignored local dump). Gold is never
written into the stored user turn. Rows that already exist are skipped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.compat.math_grading import grade_answer_verl
from training.rl_data.generate_sft_solutions import _make_sft_row, is_acceptable_solution
from training.rl_data.screen_training_data import gold_fit_drop_reason, prompt_drop_reason

DST = ROOT / "data/rl/sft_solutions.jsonl"
SRC = ROOT / "data/rl/swift_prompts.jsonl"

MANUAL = {
    "33658": r"""The surface is smooth, so the only horizontal forces after \(F\) is applied are \(F\) and the spring force. Equilibrium of the driven oscillator is at
\[
x_{\mathrm{eq}}=\frac{F}{k}.
\]
With \(\omega=\sqrt{k/m}\) and the initial conditions \(x(0)=0\), \(\dot x(0)=0\),
\[
x(t)=\frac{F}{k}(1-\cos\omega t),\qquad
\dot x(t)=\frac{F}{k}\omega\sin\omega t.
\]
The mass is instantaneously at rest when \(\sin\omega t=0\), i.e. \(\omega t=n\pi\) for integer \(n\ge 1\). At the even values \(n=2\ell\),
\[
x=0,\qquad \dot x=0.
\]
If \(F\) is then removed, the net force remains zero and the mass stays at rest. The corresponding times are
\[
t=n\cdot 2\pi\sqrt{\frac{m}{k}},\qquad n=1,2,\ldots
\]

\boxed{t=n\cdot 2\pi\sqrt{m/k}}
""",
    "156_355": r"""The orbital radius \(r\) is fixed by the Coulomb force in the Bohr model. With the weak field \(\mathbf B\) along the line of sight and the electron circulating counterclockwise when one looks *along* \(\mathbf B\), the Lorentz force \(q(\mathbf v\times\mathbf B)\) on the electron (\(q=-e\)) is radially *outward*.

Radial balance is then
\[
\frac{ke^{2}}{r^{2}}-evB=m\omega^{2}r.
\]
Compared with the field-free case \(ke^{2}/r^{2}=m\omega_{0}^{2}r\), the centripetal requirement is smaller, so \(\omega<\omega_{0}\). The angular velocity therefore decreases.

\boxed{Decrease}
""",
    "137_437": r"""The bulk modulus is
\[
K=-V\frac{dP}{dV}=\rho\frac{dP}{d\rho},
\]
hence
\[
\frac{d\rho}{\rho}=\frac{dP}{K}.
\]
Integrating from the surface (\(\rho=\rho_{0}\), \(P=0\)) to depth \(x\) with constant \(K\) gives
\[
\ln\frac{\rho(x)}{\rho_{0}}=\frac{P(x)}{K},
\]
and for \(P\ll K\) this is the linear relation
\[
\rho(x)=\rho_{0}\left(1+\frac{P(x)}{K}\right).
\]

\boxed{\rho(x)=\rho_0(1+P(x)/K)}
""",
    "41461": r"""Linear drag \(\mathbf f=-b\mathbf v\) dissipates kinetic energy independently of \(\mathbf B\), so the *path length* to rest is the same in all three cases,
\[
s_{\max}=\frac{mv_{0}}{b}.
\]
Without \(\mathbf B\) the path is straight, hence \(s_{\max}=10\,\mathrm{cm}\). With a perpendicular field the trajectory is a logarithmic spiral; the straight-line distance from the entry point to the rest point is
\[
d=\frac{v_{0}}{\sqrt{\gamma^{2}+\omega^{2}}},\qquad \gamma=\frac{b}{m},\quad\omega=\frac{qB}{m}.
\]
The field-free case is \(d_{0}=v_{0}/\gamma=10\,\mathrm{cm}\). With the original field \(d=6\,\mathrm{cm}\), so
\[
\omega^{2}=\left(\frac{v_{0}}{6}\right)^{2}-\left(\frac{v_{0}}{10}\right)^{2}.
\]
Halving \(B\) replaces \(\omega\) by \(\omega/2\):
\[
d'=\frac{v_{0}}{\sqrt{\gamma^{2}+(\omega/2)^{2}}}=\frac{30}{\sqrt{13}}\,\mathrm{cm}\approx 8.3\,\mathrm{cm}.
\]

\boxed{8.3 cm}
""",
    "118_226": r"""The filament is a thin spherical shell of radius \(R=3\,\mathrm{cm}\) and thickness \(t=0.5\,\mathrm{mm}\). Current enters and leaves through two opposite circular contacts of radius \(a=0.01\,\mathrm{mm}\).

For a thin spherical shell the resistance between two small opposite electrodes is
\[
R_{\Omega}=\frac{\rho}{\pi t}\ln\frac{2R}{a}.
\]
Substituting \(\rho=0.050\,\Omega\cdot\mathrm{m}\), \(t=5.0\times 10^{-4}\,\mathrm{m}\), \(R=0.030\,\mathrm{m}\) and \(a=1.0\times 10^{-5}\,\mathrm{m}\),
\[
\ln\frac{2R}{a}=\ln 6000\approx 8.70,\qquad
\frac{\rho}{\pi t}\approx 31.8\,\Omega,
\]
hence \(R_{\Omega}\approx 277\,\Omega\).

\boxed{277}
""",
    "183_734": r"""Photon momentum arrives at rate \(P/c\). Because the coating eliminates partial reflection, the beam is either transmitted through a face or totally internally reflected. The laser must stay parallel to a cube face, so the interior ray lies in a plane parallel to that face.

The largest possible deflection of the exit beam is \(180^\circ\), which transfers momentum \(2P/c\) to the cube, but only if TIR can reverse the ray. That requires the incidence angle on an interior face to reach \(45^\circ\) with \(\sin\theta_c=1/n\le\sin 45^\circ\), i.e. \(n\ge\sqrt{2}\). Then
\[
F_{\max}=\frac{2P}{c}.
\]
If \(n<\sqrt{2}\), the maximum interior incidence is \(\theta_c=\arcsin(1/n)\). Geometry in the allowed plane gives a maximum momentum transfer
\[
\Delta p_{\mathrm{ph}}=\frac{2P}{c}\sqrt{n^{2}-1},
\]
hence
\[
F_{\max}=\frac{2P}{c}\sqrt{n^{2}-1}.
\]

\boxed{F = \begin{cases}2 \frac{P}{c} \sqrt{n^{2}-1}, & \text { if } n<\sqrt{2} \\ 2 \frac{P}{c}, & \text { otherwise }\end{cases}}
""",
    "148_187": r"""On the ground the thrower's speed relative to himself is \(v\). The same relative speed is used on the boat. Take the throwing direction as positive. Initially student (\(M\)), boat (\(2M\)) and two balls (\(2m\)) are at rest.

After the first throw the remaining mass is \(3M+m\). Momentum conservation with ball velocity \(V_{1}+v\) gives
\[
V_{1}=-\frac{mv}{3M+2m}.
\]
After the second throw the remaining mass is \(3M\),
\[
V_{2}=V_{1}-\frac{mv}{3M+m}=-\frac{3m(2M+m)}{(3M+2m)(3M+m)}v.
\]
The boat (and student) therefore recede at speed
\[
\frac{3m(2M+m)}{(3M+2m)(3M+m)}v.
\]

\boxed{\dfrac{3m(2M+m)}{(3M+2m)(3M+m)}v}
""",
    "220_981": r"""The two samples have the same volume \(V\). Mole numbers are \(n_{1}=p_{1}V/(RT_{1})\) and \(n_{2}=p_{2}V/(RT_{2})\). Mixing is adiabatic as a whole and the total volume is \(2V\). Energy conservation for an ideal gas with constant \(C_{V}\) is
\[
n_{1}C_{V}T_{1}+n_{2}C_{V}T_{2}=(n_{1}+n_{2})C_{V}T,
\]
so
\[
T=\frac{n_{1}T_{1}+n_{2}T_{2}}{n_{1}+n_{2}}=\frac{(p_{1}+p_{2})T_{1}T_{2}}{p_{1}T_{2}+p_{2}T_{1}}.
\]
(The mixed pressure is then \(P=\frac12(p_{1}T/T_{1}+p_{2}T/T_{2})\).)

\boxed{T=\frac{(p_1+p_2)T_1 T_2}{p_1 T_2+p_2 T_1}}
""",
    "181_176": r"""Let the resistance on a mass \(\mu\) be \(\alpha\mu g\). Uniform motion of the whole train requires a traction \(F=\alpha Mg\).

At detachment both parts still have speed \(V\). The carriage (mass \(m\)) has only resistance, so its deceleration is \(\alpha g\) from that instant.

The locomotive (mass \(M-m\)) still has traction \(F\) for a further time \(t\). Its net force is \(\alpha mg\), so it *accelerates* at \(\alpha mg/(M-m)\) during those \(t\) seconds, reaching
\[
V'=V+\frac{\alpha mg}{M-m}t.
\]
Then the throttle is closed and both parts decelerate at \(\alpha g\). The extra running time of the locomotive relative to the carriage is the extra time needed to kill the additional speed \(V'-V\), plus the time \(t\) during which the carriage was already braking while the locomotive was not:
\[
\tau=\frac{V'-V}{\alpha g}+t=\frac{M}{M-m}t.
\]

\boxed{\tau=\frac{M}{M-m}t}
""",
    "117_138": r"""At the apex the rocket is instantaneously at rest, so the centre of mass of the three equal fragments has zero velocity just after the explosion and thereafter falls from rest.

Let downward be positive. Vertical momentum conservation gives \(u_{1}+2u_{2}=0\), so the fragment that falls “straight down” and the other pair have vertical speeds related by \(u_{1}=-2u_{2}\).

The drop from height \(h\) then reads
\[
h=u_{1}t_{1}+\tfrac12 gt_{1}^{2}=u_{2}t_{2}+\tfrac12 gt_{2}^{2}.
\]
Eliminating \(u_{2}\) yields
\[
h=\frac12 g t_{1}t_{2}\cdot\frac{2t_{2}+t_{1}}{2t_{1}+t_{2}}.
\]

\boxed{h=\frac12 g t_1 t_2\cdot\frac{2t_2+t_1}{2t_1+t_2}}
""",
    "233_579": r"""The gas is uniform with number density \(N/V\). The mean potential energy of one interior atom is
\[
\langle U\rangle=\frac{N}{V}\int u(r)\,dV
=\frac{N}{V}\int_{d}^{\infty}(-\epsilon)\left(\frac{d}{r}\right)^{6}4\pi r^{2}\,dr.
\]
The integral is
\[
-4\pi\epsilon d^{6}\int_{d}^{\infty}r^{-4}\,dr=-\frac{4\pi\epsilon d^{3}}{3}.
\]
With the given \(a'=2\pi d^{3}\epsilon/3\) one has \(4\pi\epsilon d^{3}/3=2a'\), so
\[
\langle U\rangle=-2\frac{a'N}{V}.
\]

\boxed{-2 a' N/V}
""",
    "167_1020": r"""Each glass–air interface reflects with probability \(r\) and transmits with probability \(1-r\). The thickness \(a=(100.25\lambda)/n\) makes the two-surface stack equivalent to a Fabry–Pérot plate whose intensity reflectance (summing the coherent multiple-reflection series) is
\[
R=\frac{4r}{(1+r)^{2}}.
\]
The plate is lossless, so transmitted light carries away the same momentum it brought in. Only the reflected fraction reverses photon momentum, and the force on the plate is
\[
F_{c}=\frac{2RP}{c}=\frac{8Pr}{c(1+r)^{2}}.
\]

\boxed{\dfrac{8Pr}{c(1+r)^{2}}}
""",
}


def main() -> None:
    if not SRC.is_file():
        print(json.dumps({"error": f"missing prompt dump {SRC}", "added": 0}, ensure_ascii=False))
        raise SystemExit(2)
    by_id = {}
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        by_id[str(row.get("sample_id") or "")] = row
    done = set()
    if DST.is_file():
        for line in DST.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(str(json.loads(line).get("sample_id") or ""))
    n_ok = 0
    n_fail = 0
    DST.parent.mkdir(parents=True, exist_ok=True)
    with DST.open("a", encoding="utf-8") as f:
        for sid, text in MANUAL.items():
            if sid in done:
                print("skip existing", sid)
                continue
            src = by_id.get(sid)
            if src is None:
                print("missing prompt", sid)
                n_fail += 1
                continue
            if prompt_drop_reason(src):
                print("screen drop", sid, prompt_drop_reason(src))
                n_fail += 1
                continue
            gold = str(src.get("solution") or "")
            if gold_fit_drop_reason(text) or not is_acceptable_solution(text, gold, min_chars=200):
                ok = grade_answer_verl(text, gold)
                print("REJECT", sid, "grade", ok, "len", len(text))
                n_fail += 1
                continue
            rec = _make_sft_row(src, text.strip(), hint_gold=False)
            rec["generator"] = "manual"
            rec["manual_edit"] = "hand_label"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
            print("added", sid)
    print(json.dumps({"added": n_ok, "failed": n_fail}))


if __name__ == "__main__":
    main()
