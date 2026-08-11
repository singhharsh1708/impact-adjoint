"""E2: material calibration from observed impact positions.

Truth (e*, mu*) generates impact positions on bumpy terrain; observations are
those positions plus Gaussian noise. Gradient descent through the contact-sim
Tesseract's saltation-aware VJP recovers (e, mu). This is the Track-4-flavored
inverse problem: the observable (where the ball actually hit) exists only
because of events, so no smooth-relaxation surrogate can express it.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

import jax
import jax.numpy as jnp
import optax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parent.parent

FIXED = {
    "v0": np.array([2.0, 0.5]), "y0": 1.0,
    "amp": np.array([0.2, 0.1, 0.15]), "ctr": np.array([1.0, 2.5, 4.0]),
    "wid": np.array([0.5, 0.4, 0.6]),
    "drag": 0.0, "t_final": 2.0, "dt": 1e-3, "n_samples": 0, "v_stop": 1e-4,
}
E_TRUE, MU_TRUE = 0.7, 0.1
NOISE = 0.005
SEED = 7


def main():
    sim = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")

    truth = sim.apply({**FIXED, "e": E_TRUE, "mu": MU_TRUE})
    nev = int(truth["n_events"])
    # Fit the first NFIT impacts: impact 1 is (e, mu)-independent (launch is
    # fixed), impacts 2-3 constrain both parameters, and every plausible
    # parameter value produces at least NFIT impacts (padded entries are
    # zeros and must never enter the residual).
    NFIT = 3
    assert nev >= NFIT
    rng = np.random.default_rng(SEED)
    obs = np.asarray(truth["impact_x"])[:NFIT] + rng.normal(0.0, NOISE, NFIT)
    print(f"observed first {NFIT} of {nev} impacts (noise sigma={NOISE}):", obs.round(4))

    def loss_fn(params):
        res = apply_tesseract(sim, {**FIXED, "e": params["e"], "mu": params["mu"]})
        r = res["impact_x"][:NFIT] - jnp.asarray(obs)
        return jnp.sum(r**2), res["n_events"]

    params = {"e": jnp.asarray(0.5), "mu": jnp.asarray(0.3)}
    opt = optax.adam(learning_rate=0.05)
    state = opt.init(params)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    for it in range(80):
        (loss, nev_it), grads = grad_fn(params)
        assert int(nev_it) >= NFIT, f"iter {it}: only {int(nev_it)} impacts"
        if it % 10 == 0:
            print(f"iter {it:3d}  loss {float(loss):.6e}  e {float(params['e']):.4f}  mu {float(params['mu']):.4f}")
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
        params["e"] = jnp.clip(params["e"], 0.05, 0.95)
        params["mu"] = jnp.clip(params["mu"], 0.0, 0.6)

    e_hat, mu_hat = float(params["e"]), float(params["mu"])
    print(f"\nrecovered e  = {e_hat:.4f}  (truth {E_TRUE}, err {abs(e_hat - E_TRUE):.4f})")
    print(f"recovered mu = {mu_hat:.4f}  (truth {MU_TRUE}, err {abs(mu_hat - MU_TRUE):.4f})")
    # gate at ~3x the noise-propagated posterior sd (see e2b: sd ~ 0.008/0.010)
    assert abs(e_hat - E_TRUE) < 0.03 and abs(mu_hat - MU_TRUE) < 0.04, "calibration failed"
    print("E2 PASSED: (e, mu) recovered from noisy impact observations via saltation gradients")


if __name__ == "__main__":
    main()
