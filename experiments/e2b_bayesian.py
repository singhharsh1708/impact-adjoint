"""E2b: Bayesian calibration. NUTS posterior over (e, mu) through the Tesseract.

NumPyro's NUTS sampler needs a JAX-differentiable log-density; the contact
solver sits inside it as a Tesseract, its saltation VJP feeding HMC's momentum
updates. Posterior mean +/- sd quantifies what three noisy impact observations
actually pin down.
"""

import time
from pathlib import Path

import numpy as np

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.diagnostics import gelman_rubin
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
        # e capped at 0.85: beyond it this configuration can exceed the event
        # capacity within t_final, where the padded-impact likelihood is
        # meaningless (see E6's failure edge for the physical regime change)
        e = numpyro.sample("e", dist.Uniform(0.3, 0.85))
        mu = numpyro.sample("mu", dist.Uniform(0.0, 0.5))
        numpyro.sample("obs", dist.Normal(forward(e, mu), NOISE), obs=jnp.asarray(obs))

    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=500, num_samples=1000,
                num_chains=2, chain_method="sequential", progress_bar=False)
    t0 = time.perf_counter()
    mcmc.run(jax.random.PRNGKey(0), extra_fields=("diverging", "num_steps"))
    wall_s = time.perf_counter() - t0
    mcmc.print_summary()
    s = mcmc.get_samples()
    extra = mcmc.get_extra_fields()
    n_div = int(np.asarray(extra["diverging"]).sum())
    # one leapfrog step = one apply plus one saltation VJP through the container
    n_leapfrog = int(np.asarray(extra["num_steps"]).sum())
    grouped = mcmc.get_samples(group_by_chain=True)
    r_hat = {k: float(gelman_rubin(np.asarray(v))) for k, v in grouped.items()}
    e_s, mu_s = np.asarray(s["e"]), np.asarray(s["mu"])
    e_mean, e_sd = float(e_s.mean()), float(e_s.std())
    mu_mean, mu_sd = float(mu_s.mean()), float(mu_s.std())

    def ci(x, lo, hi):
        return float(np.quantile(x, lo)), float(np.quantile(x, hi))

    e68, e95 = ci(e_s, 0.16, 0.84), ci(e_s, 0.025, 0.975)
    mu68, mu95 = ci(mu_s, 0.16, 0.84), ci(mu_s, 0.025, 0.975)
    print(f"divergences: {n_div}")
    print(f"posterior e  = {e_mean:.4f} +/- {e_sd:.4f}  68% CI {np.round(e68,4)}  95% CI {np.round(e95,4)}  (truth {E_TRUE})")
    print(f"posterior mu = {mu_mean:.4f} +/- {mu_sd:.4f}  68% CI {np.round(mu68,4)}  95% CI {np.round(mu95,4)}  (truth {MU_TRUE})")
    assert n_div == 0, f"{n_div} divergent transitions"
    assert e95[0] <= E_TRUE <= e95[1], "truth outside 95% CI for e"
    assert mu95[0] <= MU_TRUE <= mu95[1], "truth outside 95% CI for mu"
    print(f"r_hat: e {r_hat['e']:.4f}  mu {r_hat['mu']:.4f}")
    print(f"sampling wall time {wall_s:.1f} s over {n_leapfrog} leapfrog steps "
          f"(one apply plus one VJP each)")
    np.savez(ROOT / "experiments" / "e2b_posterior.npz", e=e_s, mu=mu_s,
             n_divergences=n_div, r_hat_e=r_hat["e"], r_hat_mu=r_hat["mu"],
             n_draws=len(e_s), n_chains=2, n_warmup=500,
             wall_s=wall_s, n_leapfrog=n_leapfrog)
    print("E2b PASSED: 2-chain NUTS through the Tesseract, 0 divergences, truth inside both 95% CIs")


def main():
    # NumPyro jits the NUTS step, and jitted JAX callbacks may run off the main
    # thread, which deadlocks the in-process juliacall runtime. The
    # containerized solver keeps Julia in its own process, so HMC's callbacks
    # are plain HTTP calls and threading is a non-issue.
    out = ROOT / ".tessout"
    out.mkdir(exist_ok=True)
    with Tesseract.from_image("contact-sim", output_path=out) as sim:
        run(sim)


if __name__ == "__main__":
    main()
