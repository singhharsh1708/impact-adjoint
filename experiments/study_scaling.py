"""Cost scaling of apply and VJP with the number of design parameters.

The sensitivities are forward-variational, so a VJP integrates one column per
parameter and its cost should grow linearly in n_params while apply stays
flat. Measuring this is what justifies the claim that a reverse-mode
saltation adjoint is the right extension at large design dimension, rather
than an aesthetic preference.

Writes experiments/scaling_result.npz.
"""

import time
from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

from tesseract_core import Tesseract

ROOT = Path(__file__).parent.parent
API = ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py"

NBS = [3, 6, 12, 24, 48, 96, 192]
REPS = 15
COT = {"qf": np.array([1.0, 0.0, 0.0, 0.0])}
ALL_IN = {"v0", "y0", "e", "mu", "amp", "ctr", "wid"}


def cfg_for(nb):
    return {
        "v0": np.array([2.0, 0.5]), "y0": 1.0, "e": 0.7, "mu": 0.1,
        "amp": np.full(nb, 0.6 / nb), "ctr": np.linspace(0.4, 5.0, nb),
        "wid": np.full(nb, 0.25), "drag": 0.0, "t_final": 2.0,
        "dt": 1e-3, "n_samples": 0,
    }


def best_ms(fn, reps=REPS):
    """Minimum over repeats. For timing under a noisy load the minimum is the
    robust estimator: noise only ever adds time, so the floor is the signal."""
    fn()  # warm
    ts = []
    for _ in range(reps):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1e3)
    return float(np.min(ts))


def main():
    t = Tesseract.from_tesseract_api(API)
    n_params, t_apply, t_vjp, n_events = [], [], [], []

    for nb in NBS:
        cfg = cfg_for(nb)
        r = t.apply(cfg)
        p = 5 + 3 * nb
        ta = best_ms(lambda: t.apply(cfg))
        tv = best_ms(lambda: t.vector_jacobian_product(
            cfg, vjp_inputs=ALL_IN, vjp_outputs={"qf"}, cotangent_vector=COT))
        n_params.append(p); t_apply.append(ta); t_vjp.append(tv)
        n_events.append(int(r["n_events"]))
        print(f"nb={nb:3d}  params={p:4d}  apply {ta:6.2f} ms   vjp {tv:7.2f} ms   "
              f"ratio {tv/ta:5.1f}x   events {int(r['n_events'])}")

    n_params = np.array(n_params, float)
    t_apply = np.array(t_apply); t_vjp = np.array(t_vjp)

    # The cost model is affine, not a power law: one forward solve worth of
    # work plus one variational column per parameter, on top of a fixed
    # per-call overhead (schema validation, marshalling). Fit t = a + b n.
    b_vjp, a_vjp = np.polyfit(n_params, t_vjp, 1)
    b_apply, a_apply = np.polyfit(n_params, t_apply, 1)
    pred = a_vjp + b_vjp * n_params
    r2 = 1 - np.sum((t_vjp - pred) ** 2) / np.sum((t_vjp - t_vjp.mean()) ** 2)
    slope_vjp, slope_apply = float(b_vjp), float(b_apply)
    print(f"\nVJP  cost model: {a_vjp:.2f} ms + {b_vjp*1000:.1f} us per parameter  (R2 = {r2:.3f})")
    print(f"apply cost model: {a_apply:.2f} ms + {b_apply*1000:.1f} us per parameter")
    print(f"marginal cost ratio, VJP vs apply, per parameter: {b_vjp/max(b_apply,1e-9):.0f}x")
    print(f"extrapolated VJP at 1000 parameters: {(a_vjp + b_vjp*1000)/1000:.2f} s")

    np.savez(ROOT / "experiments" / "scaling_result.npz",
             n_params=n_params, t_apply=t_apply, t_vjp=t_vjp,
             n_events=np.array(n_events), b_vjp=b_vjp, a_vjp=a_vjp,
             b_apply=b_apply, a_apply=a_apply, r2_vjp=r2)

    assert r2 > 0.95, f"affine cost model should fit well, got R2 = {r2:.3f}"
    assert b_vjp > 5 * max(b_apply, 1e-9), "VJP per-parameter cost should dominate apply's"
    print("\nSCALING STUDY PASSED: VJP cost is affine in parameter count with a "
          f"marginal {b_vjp*1000:.0f} us per parameter, while apply is essentially flat. "
          "This is the regime where a reverse-mode saltation adjoint would pay off.")


if __name__ == "__main__":
    main()
