"""Gradient accuracy against finite differences, swept over the FD step.

The classic diagnostic: central differences have truncation error O(h^2) and
roundoff error O(eps/h), so agreement with a correct analytic gradient traces
a V in h, bottoming near h ~ (eps)^(1/3). A wrong analytic gradient produces a
flat floor instead, because the disagreement is dominated by the error in the
gradient rather than by the FD step.

Run over four inputs of different character (a launch velocity component, the
restitution, a terrain amplitude, a terrain centre) on a 4-impact trajectory.

Writes experiments/gradient_accuracy_result.npz.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

from tesseract_core import Tesseract

ROOT = Path(__file__).parent.parent
API = ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py"

BASE = {
    "v0": np.array([2.0, 0.5]), "y0": 1.0, "e": 0.7, "mu": 0.1,
    "amp": np.array([0.2, 0.1, 0.15]), "ctr": np.array([1.0, 2.5, 4.0]),
    "wid": np.array([0.5, 0.4, 0.6]),
    "drag": 0.0, "t_final": 2.0, "dt": 1e-3, "n_samples": 0,
}
HS = np.logspace(-11, -2, 19)
PROBES = [("v0", 1), ("e", 0), ("amp", 0), ("ctr", 1)]


def perturbed(name, idx, h):
    c = {k: (np.array(v, dtype=float) if isinstance(v, np.ndarray) else v) for k, v in BASE.items()}
    val = np.atleast_1d(np.array(c[name], dtype=float)).copy()
    val[idx] += h
    c[name] = val if val.size > 1 else float(val[0])
    return c


def main():
    t = Tesseract.from_tesseract_api(API)
    r0 = t.apply(BASE)
    assert int(r0["n_events"]) == 4 and int(r0["status"]) == 0

    jac = t.jacobian(BASE, jac_inputs={n for n, _ in PROBES}, jac_outputs={"qf"})

    curves, analytic = {}, {}
    for name, idx in PROBES:
        col = np.asarray(jac["qf"][name])
        g = float(col[0, idx]) if col.ndim == 2 else float(col[0])
        analytic[f"{name}{idx}"] = g
        rel = []
        for h in HS:
            xp = float(np.asarray(t.apply(perturbed(name, idx, h))["qf"])[0])
            xm = float(np.asarray(t.apply(perturbed(name, idx, -h))["qf"])[0])
            fd = (xp - xm) / (2 * h)
            rel.append(abs(fd - g) / (abs(g) + 1e-30))
        curves[f"{name}{idx}"] = np.array(rel)
        best = np.nanmin(rel)
        print(f"{name}[{idx}]: analytic d x(T)/d = {g:+.8f}   best FD agreement {best:.2e} "
              f"at h = {HS[int(np.nanargmin(rel))]:.1e}")

    np.savez(ROOT / "experiments" / "gradient_accuracy_result.npz",
             hs=HS, **{f"rel_{k}": v for k, v in curves.items()},
             **{f"an_{k}": v for k, v in analytic.items()})

    worst_best = max(float(np.nanmin(v)) for v in curves.values())
    assert worst_best < 1e-6, f"some probe never agreed with FD better than {worst_best:.1e}"
    # a correct gradient must show the V: error at tiny h (roundoff) and at
    # large h (truncation) both exceed the minimum by orders of magnitude
    for k, v in curves.items():
        assert v[0] > 50 * np.nanmin(v), f"{k}: no roundoff branch, FD curve is flat"
        assert v[-1] > 50 * np.nanmin(v), f"{k}: no truncation branch, FD curve is flat"
    print(f"\nGRADIENT ACCURACY PASSED: all probes bottom out below {worst_best:.1e} "
          "and show both the roundoff and truncation branches")


if __name__ == "__main__":
    main()
