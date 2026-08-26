"""E5b: Adam over a learning-rate grid on the ensemble objective.

The gradient-free control (e5b_gradient_free_control.py) gave CMA-ES a 3x3
grid of initial step sizes and seeds while Adam stood on the single published
learning rate 0.004, which was never swept. CMA-ES then beat the published
design on every one of its nine runs. Deciding the comparison there would be
deciding it by the protocol: study_optimizers.py exists because "tuning only
the baseline, or only ours, would decide the outcome by the protocol rather
than by the method".

So Adam gets the same treatment here, on the same objective, same warm start,
same iteration budget. Adam is deterministic given a learning rate (the
training ensemble is drawn once from a fixed seed and the warm start is fixed),
so the grid is over learning rates alone.

This is run as a sweep and reported as a sweep whichever way it comes out. It
is not a search for a learning rate that rescues the published claim: the grid
is fixed before the run, brackets the published value, and every point is
reported.
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
from e5_separator import (  # noqa: E402
    AMP_MAX, BIN_PET, BIN_RUBBER, CTR, FIXED, WID,
)
from e5b_robust_separator import N_ITERS, N_TEST, N_TRAIN, draw_ensemble  # noqa: E402

ROOT = Path(__file__).parent.parent
X_MID = 0.5 * (BIN_RUBBER + BIN_PET)

# Fixed before the run. Brackets the published 0.004 on both sides.
LRS = (0.001, 0.002, 0.004, 0.01, 0.02)


def main():
    sim = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    score = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "score_target" / "tesseract_api.py")

    d5 = np.load(ROOT / "experiments" / "e5_result.npz")
    point_amp = np.asarray(d5["adam_amp"], dtype=float)
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
        out = []
        for v0, e, bin_x in ensemble:
            r = sim.apply({**FIXED, "v0": np.asarray(v0), "e": float(e),
                           "amp": np.asarray(amp), "ctr": CTR, "wid": WID})
            x = float(np.asarray(r["qf"])[0])
            out.append((X_MID - x) if bin_x == BIN_RUBBER else (x - X_MID))
        return np.asarray(out)

    grad_fn = jax.value_and_grad(loss_fn)
    runs = []

    print(f"Adam on the ensemble objective, {len(LRS)} learning rates, "
          f"{N_ITERS} iterations each (same budget as E5b)\n")

    for lr in LRS:
        amp = jnp.asarray(point_amp)
        opt = optax.adam(learning_rate=lr)
        state = opt.init(amp)
        # best of everything actually scored, including the warm start and the
        # final iterate: the defect this repo withdrew a release over
        best, best_amp = float(np.inf), np.asarray(point_amp)
        for _ in range(N_ITERS):
            val, g = grad_fn(amp)
            v = float(val)
            if v < best:
                best, best_amp = v, np.asarray(amp)
            upd, state = opt.update(g, state)
            amp = jnp.clip(optax.apply_updates(amp, upd), 0.0, AMP_MAX)
        final = float(loss_fn(amp))
        if final < best:
            best, best_amp = final, np.asarray(amp)

        m = margins(best_amp, test)
        k, n = int((m > 0).sum()), len(m)
        p5 = float(np.percentile(m, 5))
        runs.append({"lr": lr, "loss": best, "amp": best_amp, "margins": m,
                     "correct": k, "n": n, "p5": p5})
        print(f"  lr={lr:6.4f}  loss {best:.4e}   held-out {k}/{n}   p5 {p5:+.4f} m")

    best_run = min(runs, key=lambda r: r["loss"])
    med = float(np.median([r["loss"] for r in runs]))
    print(f"\n  best  lr={best_run['lr']}  loss {best_run['loss']:.4e}  "
          f"p5 {best_run['p5']:+.4f} m")
    print(f"  median across the grid: {med:.4e}")

    ctrl = np.load(ROOT / "experiments" / "e5b_control_result.npz")
    cma_best, cma_med = float(ctrl["cma_es_loss"]), float(ctrl["cma_median_loss"])
    print(f"\n  against the gradient-free control:")
    print(f"    CMA-ES best   {cma_best:.4e}  (median {cma_med:.4e})")
    print(f"    Adam   best   {best_run['loss']:.4e}  (median {med:.4e})")
    verdict = "Adam" if best_run["loss"] < cma_best else "CMA-ES"
    print(f"    best-of-grid winner on the ensemble objective: {verdict}")

    assert all(np.isfinite(r["loss"]) for r in runs), "non-finite objective"
    np.savez(
        ROOT / "experiments" / "e5b_adam_sweep.npz",
        lrs=np.asarray(LRS), losses=np.asarray([r["loss"] for r in runs]),
        p5s=np.asarray([r["p5"] for r in runs]),
        correct=np.asarray([r["correct"] for r in runs]),
        n_test=runs[0]["n"], median_loss=med, iters=N_ITERS,
        best_lr=best_run["lr"], best_loss=best_run["loss"],
        best_p5=best_run["p5"], best_correct=best_run["correct"],
        best_amp=best_run["amp"], best_margins=best_run["margins"],
    )
    print("\nwrote e5b_adam_sweep.npz")


if __name__ == "__main__":
    main()
