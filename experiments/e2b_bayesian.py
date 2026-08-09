"""E2b: Bayesian calibration — NUTS posterior over (e, mu) through the Tesseract.

NumPyro's NUTS sampler needs a JAX-differentiable log-density; the contact
solver sits inside it as a Tesseract, its saltation VJP feeding HMC's momentum
updates. Posterior mean +/- sd quantifies what three noisy impact observations
actually pin down.
"""

from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)
numpyro.set_host_device_count(1)

ROOT = Path(__file__).parent.parent

FIXED = {
    "v0": np.array([2.0, 0.5]), "y0": 1.0,
    "amp": np.array([0.2, 0.1, 0.15]), "ctr": np.array([1.0, 2.5, 4.0]),
    "wid": np.array([0.5, 0.4, 0.6]),
    "drag": 0.0, "t_final": 2.0, "dt": 1e-3, "n_samples": 0, "v_stop": 1e-4,
}
E_TRUE, MU_TRUE = 0.7, 0.1
NOISE = 0.005
NFIT = 3
SEED = 7


def run(sim):
    truth = sim.apply({**FIXED, "e": E_TRUE, "mu": MU_TRUE})
    rng = np.random.default_rng(SEED)
    obs = np.asarray(truth["impact_x"])[:NFIT] + rng.normal(0.0, NOISE, NFIT)
    print(f"observations ({NFIT} impacts, sigma={NOISE}):", obs.round(4))

    def forward(e, mu):
        res = apply_tesseract(sim, {**FIXED, "e": e, "mu": mu})
        return res["impact_x"][:NFIT]

    def model():
        e = numpyro.sample("e", dist.Uniform(0.3, 0.95))
        mu = numpyro.sample("mu", dist.Uniform(0.0, 0.5))
        numpyro.sample("obs", dist.Normal(forward(e, mu), NOISE), obs=jnp.asarray(obs))

    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=150, num_samples=300,
                num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0))
    s = mcmc.get_samples()
    e_mean, e_sd = float(jnp.mean(s["e"])), float(jnp.std(s["e"]))
    mu_mean, mu_sd = float(jnp.mean(s["mu"])), float(jnp.std(s["mu"]))
    n_div = int(mcmc.get_extra_fields().get("diverging", jnp.zeros(1)).sum()) if mcmc.get_extra_fields() else 0

    print(f"posterior e  = {e_mean:.4f} +/- {e_sd:.4f}   (truth {E_TRUE})")
    print(f"posterior mu = {mu_mean:.4f} +/- {mu_sd:.4f}   (truth {MU_TRUE})")
    assert abs(e_mean - E_TRUE) < 3 * max(e_sd, 1e-3), "posterior misses truth for e"
    assert abs(mu_mean - MU_TRUE) < 3 * max(mu_sd, 1e-3), "posterior misses truth for mu"
    np.savez(ROOT / "experiments" / "e2b_posterior.npz", e=np.asarray(s["e"]), mu=np.asarray(s["mu"]))
    print("E2b PASSED: NUTS posterior through the Tesseract covers the truth")


def main():
    # NumPyro jits the NUTS step, and jitted JAX callbacks may run off the main
    # thread — which deadlocks the in-process juliacall runtime. The
    # containerized solver keeps Julia in its own process, so HMC's callbacks
    # are plain HTTP calls and threading is a non-issue.
    out = ROOT / ".tessout"
    out.mkdir(exist_ok=True)
    with Tesseract.from_image("contact-sim", output_path=out) as sim:
        run(sim)


if __name__ == "__main__":
    main()
