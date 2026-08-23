"""E5b: separator design under uncertainty.

Real bounce separators process streams with scatter, not single particles.
Here the design objective is the EXPECTED landing loss over an ensemble of
particles with inlet-velocity and per-particle restitution scatter (fixed
common random numbers, so the ensemble gradient is exact). We warm-start from
the E5 point design and compare both designs on a held-out test ensemble.

Metric that matters to a plant engineer: sorting purity, the fraction of
particles landing on their own side of the bin midline.
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

import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import AMP_MAX, BIN_PET, BIN_RUBBER, CTR, E_PET, E_RUBBER, FIXED, V0, WID

ROOT = Path(__file__).parent.parent
X_MID = 0.5 * (BIN_RUBBER + BIN_PET)

V_SD = 0.05   # m/s inlet scatter, both components
E_SD = 0.03   # per-particle restitution scatter
N_TRAIN = 12  # per material
N_TEST = 100  # per material


def draw_ensemble(rng, n):
    out = []
    for e_mat, bin_x in ((E_RUBBER, BIN_RUBBER), (E_PET, BIN_PET)):
        for _ in range(n):
            v0 = V0 + rng.normal(0.0, V_SD, 2)
            e = float(np.clip(rng.normal(e_mat, E_SD), 0.05, 0.93))
            out.append((v0, e, bin_x))
    return out


def main():
    sim = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    score = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "score_target" / "tesseract_api.py")

    point_amp = np.load(ROOT / "experiments" / "e5_result.npz")["adam_amp"]
    train = draw_ensemble(np.random.default_rng(5), N_TRAIN)
    test = draw_ensemble(np.random.default_rng(17), N_TEST)

    def h_at(x, amp):
        return jnp.sum(amp * jnp.exp(-((x - CTR) ** 2) / (2 * WID**2)))

    def one(amp, v0, e, bin_x):
        res = apply_tesseract(sim, {**FIXED, "v0": v0, "e": e, "amp": amp, "ctr": CTR, "wid": WID})
        target = jnp.stack([jnp.asarray(bin_x), h_at(bin_x, amp)])
        sc = apply_tesseract(score, {"qf": res["qf"], "target": target,
                                     "weights": jnp.array([1.0, 1.0, 0.0])})
        return sc["loss"]

    def loss_fn(amp):
        return sum(one(amp, jnp.asarray(v0), e, bx) for v0, e, bx in train) / len(train)

    def margins(amp, ensemble):
        """Signed distance from the decision boundary; positive is correct."""
        out = []
        for v0, e, bx in ensemble:
            r = sim.apply({**FIXED, "v0": np.asarray(v0), "e": e, "amp": np.asarray(amp), "ctr": CTR, "wid": WID})
            x = float(np.asarray(r["qf"])[0])
            out.append((X_MID - x) if bx == BIN_RUBBER else (x - X_MID))
        return np.array(out)

    def purity(amp, ensemble):
        return float(np.mean(margins(amp, ensemble) > 0))

    # The gradient-free designs were never evaluated on the metric this
    # project says matters. The loss gap is 0.34 mm against 31 mm with bins
    # 1600 mm apart, so both may sort perfectly; the question is whether they
    # hold up under scatter.
    e5 = np.load(ROOT / "experiments" / "e5_result.npz")
    rival_purity = {}
    for rival in ("cma-es", "nelder-mead"):
        key = f"{rival}_amp"
        if key in e5.files:
            rival_purity[rival] = purity(e5[key], test)
            print(f"{rival} design: test purity {rival_purity[rival]:.3f}")

    p_point_test = purity(point_amp, test)
    print(f"point design (E5): test purity {p_point_test:.3f}  ({N_TEST}/material, "
          f"v sd {V_SD} m/s, e sd {E_SD})")

    amp = jnp.asarray(point_amp)
    opt = optax.adam(learning_rate=0.004)
    state = opt.init(amp)
    grad_fn = jax.value_and_grad(loss_fn)
    n_iters = 80
    # Keep the best point the loop actually evaluated. Returning `amp` after
    # the final update saves a vector whose objective was never computed, which
    # is the defect E5 and E4 were already fixed for.
    best, best_amp = np.inf, np.asarray(amp)
    for it in range(n_iters):
        val, g = grad_fn(amp)
        if float(val) < best:
            best, best_amp = float(val), np.asarray(amp)
        if it % 10 == 0 or it == n_iters - 1:
            print(f"iter {it:3d}  expected loss {float(val):.5f}")
        updates, state = opt.update(g, state)
        amp = jnp.clip(optax.apply_updates(amp, updates), 0.0, AMP_MAX)

    robust_amp = best_amp
    p_robust_train = purity(robust_amp, train)
    p_robust_test = purity(robust_amp, test)
    print(f"\nrobust design: train purity {p_robust_train:.3f}  test purity {p_robust_test:.3f}")
    print(f"point design : test purity {p_point_test:.3f}")

    # margin distributions for every design, so the comparison can be made on
    # separation quality rather than only on objective value
    all_margins = {n: margins(a, test) for n, a in
                   (("adam", point_amp), ("cma-es", e5["cma-es_amp"]),
                    ("nelder-mead", e5["nelder-mead_amp"]), ("ensemble", robust_amp))}
    for n, m in all_margins.items():
        print(f"  {n:12s} purity {np.mean(m > 0):.3f}  median {np.median(m):+.3f} m  "
              f"5th pct {np.percentile(m, 5):+.3f}  worst {m.min():+.3f}")

    np.savez(ROOT / "experiments" / "e5b_result.npz",
             p_cma_test=rival_purity.get("cma-es", np.nan),
             p_nm_test=rival_purity.get("nelder-mead", np.nan),
             **{f"margins_{n.replace('-', '_')}": m for n, m in all_margins.items()},
             robust_amp=robust_amp, point_amp=point_amp,
             p_point_test=p_point_test, p_robust_test=p_robust_test,
             v_sd=V_SD, e_sd=E_SD, n_test=N_TEST)

    assert p_robust_test >= p_point_test, "robust design should not be worse on held-out scatter"
    print("E5b PASSED: expected-loss design improves (or matches) held-out sorting purity")


if __name__ == "__main__":
    main()
