"""Cross-implementation reference check for contact-sim.

Re-implements the SPEC (not the Julia code) with scipy's adaptive RK45 and its
own event localization, then compares on the bumpy-terrain and drag configs,
exactly the sector (sloped contact frame, drag) the flat-terrain closed-form
oracle cannot reach. The solver's ANALYTIC Jacobian is compared against central
finite differences through THIS independent implementation.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

from scipy.integrate import solve_ivp
from tesseract_core import Tesseract

API_PATH = Path(__file__).parent.parent / "tesseracts" / "contact_sim" / "tesseract_api.py"

from tesseract_core.runtime.core import load_module_from_path

MAX_EVENTS = load_module_from_path(API_PATH).MAX_EVENTS

G = 9.81


def make_ref(ctr, wid):
    ctr = np.asarray(ctr, float)
    wid = np.asarray(wid, float)

    def h(x, amp):
        return float(np.sum(amp * np.exp(-((x - ctr) ** 2) / (2 * wid**2))))

    def hp(x, amp):
        return float(np.sum(amp * np.exp(-((x - ctr) ** 2) / (2 * wid**2)) * (-(x - ctr) / wid**2)))

    def simulate(v0, y0, e, mu, amp, drag, t_final, max_events=8):
        amp = np.asarray(amp, float)

        def rhs(t, q):
            return [q[2], q[3], -drag * q[2], -G - drag * q[3]]

        def guard(t, q):
            return q[1] - h(q[0], amp)

        guard.terminal = True
        guard.direction = -1

        q = np.array([0.0, float(y0), float(v0[0]), float(v0[1])])
        t0 = 0.0
        impacts = []
        while t0 < t_final - 1e-13:
            sol = solve_ivp(rhs, (t0, t_final), q, events=guard, rtol=1e-11, atol=1e-13, max_step=0.05)
            if sol.t_events[0].size == 0:
                return sol.y[:, -1], np.array(impacts)
            te = float(sol.t_events[0][0])
            qe = sol.y_events[0][0].copy()
            impacts.append(qe[0])
            if len(impacts) > max_events:
                raise RuntimeError("reference exceeded event budget")
            hpv = hp(qe[0], amp)
            s = np.hypot(1.0, hpv)
            n = np.array([-hpv, 1.0]) / s
            tv = np.array([1.0, hpv]) / s
            v = qe[2:]
            vpost = (-e * (v @ n)) * n + ((1 - mu) * (v @ tv)) * tv
            q = np.array([qe[0], h(qe[0], amp) + 1e-12, vpost[0], vpost[1]])
            t0 = te
        return q, np.array(impacts)

    return simulate


DIFF_INPUTS = ["v0", "y0", "e", "mu", "amp"]


def ref_fd_jacobian(simulate, base, n_events, h_fd=1e-6):
    theta0 = {
        "v0": np.array(base["v0"], float),
        "y0": float(base["y0"]),
        "e": float(base["e"]),
        "mu": float(base["mu"]),
        "amp": np.array(base["amp"], float),
    }

    def run(overrides):
        p = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in theta0.items()}
        p.update(overrides)
        qf, imp = simulate(p["v0"], p["y0"], p["e"], p["mu"], p["amp"], base["drag"], base["t_final"])
        assert imp.size == n_events, f"reference event count changed: {imp.size} != {n_events}"
        return np.concatenate([qf, imp])

    out = {}
    for name in DIFF_INPUTS:
        val = np.atleast_1d(np.asarray(theta0[name], float))
        cols = []
        for i in range(val.size):
            vp, vm = val.copy(), val.copy()
            vp[i] += h_fd
            vm[i] -= h_fd
            fp = run({name: vp if val.size > 1 else float(vp[0])})
            fm = run({name: vm if val.size > 1 else float(vm[0])})
            cols.append((fp - fm) / (2 * h_fd))
        out[name] = np.stack(cols, axis=-1)
    return out


def check(t, base, label):
    simulate = make_ref(base["ctr"], base["wid"])
    res = t.apply(base)
    nev = int(res["n_events"])
    assert int(res["status"]) == 0, f"{label}: solver truncated (status={res['status']})"

    ref_qf, ref_imp = simulate(base["v0"], base["y0"], base["e"], base["mu"], base["amp"], base["drag"], base["t_final"])
    assert ref_imp.size == nev, f"{label}: event count solver={nev} vs reference={ref_imp.size}"
    prim_err = max(
        np.max(np.abs(np.asarray(res["qf"]) - ref_qf)),
        np.max(np.abs(np.asarray(res["impact_x"])[:nev] - ref_imp)),
    )
    print(f"[{label}] n_events={nev}, primal max abs err vs scipy reference: {prim_err:.3e}")
    assert prim_err < 1e-7, f"{label}: primal mismatch {prim_err:.3e}"

    jac = t.jacobian(base, jac_inputs=set(DIFF_INPUTS), jac_outputs={"qf", "impact_x"})
    fd = ref_fd_jacobian(simulate, base, nev)
    worst = 0.0
    for name in DIFF_INPUTS:
        a_q = np.asarray(jac["qf"][name]).reshape(4, -1)
        a_i = np.asarray(jac["impact_x"][name]).reshape(MAX_EVENTS, -1)[:nev]
        analytic = np.concatenate([a_q, a_i], axis=0)
        err = np.max(np.abs(analytic - fd[name]) / (np.abs(fd[name]) + 1.0))
        worst = max(worst, err)
        np.testing.assert_allclose(analytic, fd[name], rtol=5e-4, atol=1e-6, err_msg=f"{label}: {name}")
    print(f"[{label}] analytic Jacobian vs FD-through-INDEPENDENT-implementation OK (worst scaled err {worst:.2e})")


BASE = {
    "v0": np.array([2.0, 0.5]),
    "y0": 1.0,
    "e": 0.7,
    "mu": 0.1,
    "amp": np.array([0.2, 0.1, 0.15]),
    "ctr": np.array([1.0, 2.5, 4.0]),
    "wid": np.array([0.5, 0.4, 0.6]),
    "drag": 0.0,
    "t_final": 2.0,
    "dt": 1e-3,
    "n_samples": 0,
}


def main():
    t = Tesseract.from_tesseract_api(API_PATH)
    check(t, BASE, "bumpy no-drag")
    check(t, dict(BASE, drag=0.3), "bumpy drag=0.3")
    print("\nCROSS-IMPLEMENTATION REFERENCE PASSED (scipy RK45 + independent event finder)")


if __name__ == "__main__":
    main()
