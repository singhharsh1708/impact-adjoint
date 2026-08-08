"""Validation for the contact-sim Tesseract (dev mode, no Docker).

1. Closed-form ballistic check (drag=0, pre-impact): RK4 is exact on parabolas.
2. Central finite differences vs the analytic (variational + saltation) Jacobian,
   for every differentiable input, on a multi-bounce trajectory.
3. jax.grad through tesseract-jax vs finite differences of the same scalar loss.
4. Repeat FD check with nonzero drag.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add("ForwardDiff")')

import jax
import jax.numpy as jnp
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

API_PATH = Path(__file__).parent.parent / "tesseracts" / "contact_sim" / "tesseract_api.py"

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

DIFF_INPUTS = ["v0", "y0", "e", "mu", "amp", "ctr", "wid"]


def perturbed(base, name, idx, h):
    d = {k: (np.array(v, dtype=float) if isinstance(v, np.ndarray) else v) for k, v in base.items()}
    val = np.array(d[name], dtype=float)
    if val.ndim == 0:
        d[name] = float(val) + h
    else:
        val = val.copy()
        val[idx] += h
        d[name] = val
    return d


def fd_jacobian(t, base, n_events, h=1e-6):
    """Central-difference d(qf, impact_x[:n_events])/d(theta) per input."""
    out = {}
    for name in DIFF_INPUTS:
        val = np.atleast_1d(np.array(base[name], dtype=float))
        cols = []
        for i in range(val.size):
            rp = t.apply(perturbed(base, name, i, h))
            rm = t.apply(perturbed(base, name, i, -h))
            assert int(rp["n_events"]) == n_events and int(rm["n_events"]) == n_events, (
                f"event count changed under FD perturbation of {name}[{i}]"
            )
            dq = (np.asarray(rp["qf"]) - np.asarray(rm["qf"])) / (2 * h)
            di = (np.asarray(rp["impact_x"])[:n_events] - np.asarray(rm["impact_x"])[:n_events]) / (2 * h)
            cols.append(np.concatenate([dq, di]))
        out[name] = np.stack(cols, axis=-1)  # (4 + n_events, n_in)
    return out


def check_case(t, base, label, rtol=2e-4, atol=5e-7):
    res = t.apply(base)
    nev = int(res["n_events"])
    assert nev >= 2, f"{label}: want a multi-bounce trajectory, got {nev} events"
    print(f"[{label}] n_events={nev}, qf={np.asarray(res['qf']).round(4)}")

    jac = t.jacobian(base, jac_inputs=set(DIFF_INPUTS), jac_outputs={"qf", "impact_x"})
    fd = fd_jacobian(t, base, nev)

    worst = 0.0
    for name in DIFF_INPUTS:
        a_q = np.atleast_2d(np.asarray(jac["qf"][name]).reshape(4, -1))
        a_i = np.asarray(jac["impact_x"][name]).reshape(8, -1)[:nev]
        analytic = np.concatenate([a_q, a_i], axis=0)
        err = np.max(np.abs(analytic - fd[name]) / (np.abs(fd[name]) + 1.0))
        worst = max(worst, err)
        np.testing.assert_allclose(analytic, fd[name], rtol=rtol, atol=atol, err_msg=f"{label}: {name}")
    print(f"[{label}] analytic vs FD Jacobian OK (worst scaled err {worst:.2e})")
    return nev


def main():
    t = Tesseract.from_tesseract_api(API_PATH)

    # 1. Closed-form pre-impact ballistic check.
    short = dict(BASE, t_final=0.3)
    r = t.apply(short)
    T = 0.3
    expect = np.array([2.0 * T, 1.0 + 0.5 * T - 0.5 * 9.81 * T**2, 2.0, 0.5 - 9.81 * T])
    np.testing.assert_allclose(np.asarray(r["qf"]), expect, rtol=1e-12)
    assert int(r["n_events"]) == 0
    print("[closed-form] pre-impact ballistic exact vs analytic (rtol 1e-12)")

    # 2. Multi-bounce FD check, no drag.
    nev = check_case(t, BASE, "no-drag")

    # 3. jax.grad vs FD on a scalar loss (final x + first impact x).
    def loss_fn(v0, y0, e, mu, amp, ctr, wid):
        res = apply_tesseract(
            t,
            {
                "v0": v0, "y0": y0, "e": e, "mu": mu,
                "amp": amp, "ctr": ctr, "wid": wid,
                "drag": BASE["drag"], "t_final": BASE["t_final"],
                "dt": BASE["dt"], "n_samples": BASE["n_samples"],
            },
        )
        return res["qf"][0] + res["impact_x"][0]

    args = (
        jnp.asarray(BASE["v0"]), jnp.asarray(BASE["y0"]), jnp.asarray(BASE["e"]),
        jnp.asarray(BASE["mu"]), jnp.asarray(BASE["amp"]), jnp.asarray(BASE["ctr"]),
        jnp.asarray(BASE["wid"]),
    )
    grads = jax.grad(loss_fn, argnums=tuple(range(7)))(*args)

    h = 1e-6
    for name, g in zip(DIFF_INPUTS, grads):
        val = np.atleast_1d(np.array(BASE[name], dtype=float))
        fd_g = np.zeros(val.size)
        for i in range(val.size):
            rp = t.apply(perturbed(BASE, name, i, h))
            rm = t.apply(perturbed(BASE, name, i, -h))
            lp = float(np.asarray(rp["qf"])[0] + np.asarray(rp["impact_x"])[0])
            lm = float(np.asarray(rm["qf"])[0] + np.asarray(rm["impact_x"])[0])
            fd_g[i] = (lp - lm) / (2 * h)
        np.testing.assert_allclose(np.asarray(g).reshape(-1), fd_g, rtol=2e-4, atol=5e-7, err_msg=name)
    print("[jax.grad] matches FD for all 7 differentiable inputs")

    # 4. With drag.
    check_case(t, dict(BASE, drag=0.3), "drag=0.3")

    print(f"\nALL CONTACT-SIM VALIDATIONS PASSED (multi-bounce n_events={nev})")


if __name__ == "__main__":
    main()
