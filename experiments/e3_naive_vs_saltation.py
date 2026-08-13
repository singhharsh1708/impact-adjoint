"""E3: autodiff through a time-stepper vs saltation-aware gradients.

Three ways to differentiate the same bouncing-ball program:

1. GRID-RESET (what a practitioner writes first): fixed-step RK4 scan, detect
   the guard sign change between grid points, apply the reset AT THE GRID
   POINT via jnp.where. jax.grad of this program returns exactly 0.0 for
   d x(T)/d v0y at every dt: the event-step index is piecewise-constant in the
   parameters, so autodiff differentiates a staircase. (The exact zero is
   specific to state-independent resets over flat terrain; on curved terrain
   the grid-reset gradient is nonzero and wrong instead of zero.)

2. INTERPOLATED-EVENT (the repair): locate the crossing time by a linear root
   of the guard inside the step, reset at the interpolated state, integrate
   the remainder. This recovers a converging gradient, and that is the
   point: the fix IS event-time sensitivity machinery, hand-implemented. Its
   accuracy is tied to the interpolation order and step size, and every new
   guard/reset pair needs its own hand treatment inside the autodiff program.

3. SALTATION (the Tesseract): the analytic event sensitivity, exact at any
   dt, served through the component boundary, with no event handling in the JAX
   program at all.

Truth is known in closed form on this configuration (independently validated
in validate_closed_form.py).
"""

from functools import partial
from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

import jax
import jax.numpy as jnp
from tesseract_core import Tesseract

jax.config.update("jax_enable_x64", True)

API_PATH = Path(__file__).parent.parent / "tesseracts" / "contact_sim" / "tesseract_api.py"
G = 9.81

CFG = {
    "v0": np.array([2.0, 0.5]),
    "y0": 1.0,
    "e": 0.7,
    "mu": 0.1,
    "amp": np.zeros(3),
    "ctr": np.array([1.0, 2.5, 4.0]),
    "wid": np.array([0.5, 0.4, 0.6]),
    "drag": 0.0,
    "t_final": 2.0,
    "dt": 1e-3,
    "n_samples": 0,
}


def _rk4(q, dt):
    def f(q):
        return jnp.array([q[2], q[3], 0.0, -G])

    k1 = f(q)
    k2 = f(q + dt / 2 * k1)
    k3 = f(q + dt / 2 * k2)
    k4 = f(q + dt * k3)
    return q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def _reset(q):
    return jnp.array([q[0], q[1], (1 - CFG["mu"]) * q[2], -CFG["e"] * q[3]])


@partial(jax.jit, static_argnames=("n_steps",))
def naive_final_x(v0y, n_steps):
    """Grid-reset baseline: reset applied at the grid point via jnp.where."""
    dt = CFG["t_final"] / n_steps

    def step(q, _):
        qn = _rk4(q, dt)
        crossed = (qn[1] < 0.0) & (q[1] > 0.0)
        return jnp.where(crossed, _reset(qn), qn), None

    q0 = jnp.array([0.0, CFG["y0"], 2.0, v0y])
    qf, _ = jax.lax.scan(step, q0, None, length=n_steps)
    return qf[0]


@partial(jax.jit, static_argnames=("n_steps",))
def interp_final_x(v0y, n_steps):
    """Interpolated-event repair: linear guard root inside the crossing step,
    i.e. hand-implemented first-order event-time sensitivity."""
    dt = CFG["t_final"] / n_steps

    def step(q, _):
        qn = _rk4(q, dt)
        crossed = (qn[1] < 0.0) & (q[1] > 0.0)
        alpha = jnp.clip(q[1] / (q[1] - qn[1] + 1e-300), 0.0, 1.0)
        q_hit = q + alpha * (qn - q)
        q_after = _rk4(_reset(q_hit), (1.0 - alpha) * dt)
        return jnp.where(crossed, q_after, qn), None

    q0 = jnp.array([0.0, CFG["y0"], 2.0, v0y])
    qf, _ = jax.lax.scan(step, q0, None, length=n_steps)
    return qf[0]


def main():
    t = Tesseract.from_tesseract_api(API_PATH)

    # Truth: contact-sim analytic Jacobian (independently validated to 7e-12).
    jac = t.jacobian(CFG, jac_inputs={"v0"}, jac_outputs={"qf"})
    truth = float(np.asarray(jac["qf"]["v0"])[0, 1])  # d x(T) / d v0y

    # FD through the hybrid system (via contact-sim apply) as a second witness.
    h = 1e-6
    xp = float(np.asarray(t.apply({**CFG, "v0": np.array([2.0, 0.5 + h])})["qf"])[0])
    xm = float(np.asarray(t.apply({**CFG, "v0": np.array([2.0, 0.5 - h])})["qf"])[0])
    fd_truth = (xp - xm) / (2 * h)

    print(f"d x(T)/d v0y   truth (saltation analytic): {truth:+.8f}")
    print(f"d x(T)/d v0y   FD through hybrid solver  : {fd_truth:+.8f}")
    print()
    print(f"{'n_steps':>8} {'dt':>10} {'grid-reset grad':>16} {'interp-event grad':>18} {'interp rel err':>15}")
    grad_naive = jax.grad(naive_final_x)
    grad_interp = jax.grad(interp_final_x)
    rows = []
    for n_steps in (500, 1000, 2000, 4000, 8000, 16000, 32000):
        gn = float(grad_naive(jnp.float64(0.5), n_steps))
        gi = float(grad_interp(jnp.float64(0.5), n_steps))
        rows.append((CFG["t_final"] / n_steps, gn, gi))
        print(f"{n_steps:>8} {CFG['t_final']/n_steps:>10.2e} {gn:>16.6f} {gi:>18.6f} {abs(gi-truth)/abs(truth):>15.2e}")
    np.save(Path(__file__).parent / "e3_rows.npy", np.array(rows))

    print("\ngrid-reset autodiff: exactly 0.0 at every dt (event-index staircase; the")
    print("saltation term is structurally absent). interpolated-event autodiff recovers")
    print("a converging gradient -- by hand-implementing first-order event-time")
    print("sensitivity inside the program. The Tesseract serves the exact value at any")
    print("dt with no event handling in the JAX program at all.")

    rel = abs(truth - fd_truth) / abs(truth)
    assert rel < 1e-6, f"truth witnesses disagree: {rel:.2e}"
    assert all(abs(r[1]) < 1e-15 for r in rows), "grid-reset gradient expected to be exactly 0"
    assert abs(rows[-1][2] - truth) / abs(truth) < 1e-3, "interp-event should converge toward truth"


if __name__ == "__main__":
    main()
