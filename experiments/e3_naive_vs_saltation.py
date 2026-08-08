"""E3: naive autodiff through a time-stepper vs saltation-aware gradients.

The naive approach a practitioner writes in pure JAX: fixed-step RK4 scan,
detect the guard sign change between grid points, apply the reset AT THE GRID
POINT via jnp.where. jax.grad of that program is the exact derivative of the
discrete program — but it is NOT a consistent estimator of the gradient of the
underlying hybrid system: it misses the event-time sensitivity (saltation)
term, an O(1) bias that does NOT vanish as dt -> 0.

Demonstration on the flat-terrain single/multi-bounce case where the truth is
known in closed form (validated independently in validate_closed_form.py):
sweep dt for the naive program and watch its gradient fail to converge, while
the contact-sim Tesseract reports the exact value at every dt.
"""

from functools import partial
from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add("ForwardDiff")')

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


@partial(jax.jit, static_argnames=("n_steps",))
def naive_final_x(v0y, n_steps):
    """Pure-JAX naive hybrid sim (flat terrain): reset applied at grid points."""
    dt = CFG["t_final"] / n_steps

    def rk4(q):
        def f(q):
            return jnp.array([q[2], q[3], 0.0, -G])

        k1 = f(q)
        k2 = f(q + dt / 2 * k1)
        k3 = f(q + dt / 2 * k2)
        k4 = f(q + dt * k3)
        return q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def reset(q):
        return jnp.array([q[0], q[1], (1 - CFG["mu"]) * q[2], -CFG["e"] * q[3]])

    def step(q, _):
        qn = rk4(q)
        crossed = (qn[1] < 0.0) & (q[1] > 0.0)
        return jnp.where(crossed, reset(qn), qn), None

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
    print(f"{'n_steps':>8} {'dt':>10} {'naive primal':>14} {'naive grad':>12} {'bias':>12} {'bias %':>8}")
    grad_fn = jax.grad(naive_final_x)
    solver_xT = float(np.asarray(t.apply(CFG)["qf"])[0])
    for n_steps in (500, 1000, 2000, 4000, 8000, 16000, 32000):
        g = float(grad_fn(jnp.float64(0.5), n_steps))
        primal = float(naive_final_x(jnp.float64(0.5), n_steps))
        bias = g - truth
        print(f"{n_steps:>8} {CFG['t_final']/n_steps:>10.2e} {primal:>14.6f} {g:>12.6f} {bias:>+12.6f} {100*abs(bias/truth):>7.1f}%")
    print(f"\nnaive primal converges to solver x(T)={solver_xT:.6f} as dt->0, but the naive")
    print("gradient retains an O(1) bias: the jnp.where event handling differentiates the")
    print("branch, not the event time. The saltation term is structurally absent.")

    rel = abs(truth - fd_truth) / abs(truth)
    assert rel < 1e-6, f"truth witnesses disagree: {rel:.2e}"


if __name__ == "__main__":
    main()
