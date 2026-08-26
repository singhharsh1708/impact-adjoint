"""E5b control: gradient-free optimizers on the ensemble objective.

The write-up's answer to "Nelder-Mead matches Adam on held-out purity" is that
minimising the point objective further does not buy a better separator, while
optimising the *ensemble* objective does, and that this is a gradient through
many trajectories at once. That was asserted and never measured: nobody ran a
gradient-free method on the ensemble objective. This runs the control.

Protocol, mirroring study_optimizers.py so the outcome is not decided by it:

- Same warm start as E5b (the E5 point design), same training draw, same
  held-out ensemble, same objective.
- Budget matched in particle solves. jax.value_and_grad issues an apply AND a
  vector_jacobian_product per particle, and GRAD_CHARGE is the reverse pass
  alone, so an Adam step costs (1 + GRAD_CHARGE) per particle. Same rule as
  study_optimizers.py and e5_separator.py. E5b's post-loop scoring of the final
  iterate is counted too.
- CMA-ES is swept over three initial step sizes and three seeds, because
  e5_cma_grid.py already measured a 24x spread across that grid and a single
  hardcoded configuration is the protocol failure study_optimizers.py exists to
  avoid. Nelder-Mead gets an explicit initial simplex at two scales: scipy's
  default builds it as x_i * 1.05, which from this warm start gives steps
  spanning four orders of magnitude and freezes the small amplitudes.
- Adam's side is E5b's published design at lr = 0.004, which was never swept.
  So this is one Adam configuration against a grid of gradient-free ones, and
  only the gradient-free side has a measured configuration sensitivity. That
  favours the gradient-free arm, which is the safe direction for a control
  testing our own claim.

Best-so-far traces are recorded against cumulative solves, so the cost question
can be answered afterwards without re-running.
"""

from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import (  # noqa: E402
    AMP_MAX, BIN_PET, BIN_RUBBER, CTR, FIXED, NB, WID,
)
from e5b_robust_separator import (  # noqa: E402
    N_ITERS, N_TEST, N_TRAIN, draw_ensemble,
)

ROOT = Path(__file__).parent.parent
X_MID = 0.5 * (BIN_RUBBER + BIN_PET)
OUT_PATH = ROOT / "experiments" / "e5b_control_result.npz"

GRAD_CHARGE = float(
    np.load(ROOT / "experiments" / "optimizer_benchmark.npz")["grad_charge_wall"]
)
ENSEMBLE_COST = 2 * N_TRAIN            # solver calls per objective evaluation

# Wall-clock accounting, the one generous to gradient-free.
BUDGET_SOLVES = int(round(N_ITERS * ENSEMBLE_COST * (1.0 + GRAD_CHARGE))) + ENSEMBLE_COST
MAX_EVALS = BUDGET_SOLVES // ENSEMBLE_COST
# Evaluation-count accounting, where the repo charges a gradient as 2 solves.
BUDGET_SOLVES_EVAL = N_ITERS * ENSEMBLE_COST * (1 + 2) + ENSEMBLE_COST
MAX_EVALS_EVAL = BUDGET_SOLVES_EVAL // ENSEMBLE_COST

SIGMAS = (0.01, 0.02, 0.05)
CMA_SEEDS = (3, 4, 5)
NM_SCALES = (0.01, 0.05)


def main():
    sim = Tesseract.from_tesseract_api(
        ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py"
    )

    d5 = np.load(ROOT / "experiments" / "e5_result.npz")
    d5b = np.load(ROOT / "experiments" / "e5b_result.npz")
    point_amp = np.asarray(d5["adam_amp"], dtype=float)
    robust_amp = np.asarray(d5b["robust_amp"], dtype=float)

    train = draw_ensemble(np.random.default_rng(5), N_TRAIN)
    test = draw_ensemble(np.random.default_rng(17), N_TEST)
    assert len(train) == ENSEMBLE_COST, (len(train), ENSEMBLE_COST)

    def h_at(x, amp):
        return float(np.sum(amp * np.exp(-((x - CTR) ** 2) / (2 * WID**2))))

    def landing(amp, v0, e):
        r = sim.apply({**FIXED, "v0": np.asarray(v0), "e": float(e),
                       "amp": np.asarray(amp), "ctr": CTR, "wid": WID})
        return np.asarray(r["qf"], dtype=float)

    calls = {"n": 0}
    trace = {"cur": None}

    def ensemble_loss(amp):
        """The score Tesseract's quadratic (weights [1, 1, 0]) over the draw."""
        amp = np.clip(np.asarray(amp, dtype=float), 0.0, AMP_MAX)
        total = 0.0
        for v0, e, bin_x in train:
            qf = landing(amp, v0, e)
            calls["n"] += 1
            dx = qf[0] - bin_x
            dy = qf[1] - h_at(bin_x, amp)
            total += dx * dx + dy * dy
        val = total / len(train)
        if trace["cur"] is not None:
            best = min(val, trace["cur"][-1][1]) if trace["cur"] else val
            trace["cur"].append((calls["n"], best))
        return val

    def margins(amp, ensemble):
        out = []
        for v0, e, bin_x in ensemble:
            x = landing(amp, v0, e)[0]
            out.append((X_MID - x) if bin_x == BIN_RUBBER else (x - X_MID))
        return np.asarray(out)

    def report(name, amp, loss):
        m = margins(amp, test)
        k, n = int((m > 0).sum()), len(m)
        p5 = float(np.percentile(m, 5))
        print(f"  {name:26} loss {loss:.6e}   held-out {k}/{n}   p5 {p5:+.4f} m")
        return {"loss": float(loss), "correct": k, "n": n, "p5": p5,
                "amp": np.asarray(amp, dtype=float), "margins": m}

    print(f"Budget (wall-clock accounting, generous to gradient-free):")
    print(f"  {N_ITERS} Adam iterations x {ENSEMBLE_COST} particles x "
          f"(1 + {GRAD_CHARGE:.2f}) + {ENSEMBLE_COST} final scoring")
    print(f"  = {BUDGET_SOLVES} particle solves = {MAX_EVALS} ensemble evaluations")
    print(f"  (evaluation-count accounting, gradient charged as 2, would allow "
          f"{MAX_EVALS_EVAL})")
    print(f"  the charge itself moves a few percent between runs\n")

    out = {}
    print("reference designs (recomputed, gated against e5b_result.npz):")
    out["point"] = report("warm start (E5 point)", point_amp, ensemble_loss(point_amp))
    out["adam_ensemble"] = report("E5b Adam ensemble", robust_amp, ensemble_loss(robust_amp))

    assert np.allclose(out["point"]["margins"], np.asarray(d5b["margins_adam"], float),
                       atol=1e-9), "point reference row drifted from e5b_result.npz"
    assert np.allclose(out["adam_ensemble"]["margins"],
                       np.asarray(d5b["margins_ensemble"], float), atol=1e-9), \
        "ensemble reference row drifted from e5b_result.npz"

    adam_p5 = out["adam_ensemble"]["p5"]
    adam_loss = out["adam_ensemble"]["loss"]

    print(f"\nCMA-ES on the ensemble objective, {len(SIGMAS)}x{len(CMA_SEEDS)} grid:")
    import cma
    cma_runs = []
    for sigma0 in SIGMAS:
        for seed in CMA_SEEDS:
            calls["n"] = 0
            trace["cur"] = []
            es = cma.CMAEvolutionStrategy(
                list(point_amp), sigma0,
                {"bounds": [0.0, AMP_MAX], "maxfevals": MAX_EVALS,
                 "verbose": -9, "seed": seed},
            )
            # Adam's best-evaluated tracking includes its warm start, so this
            # must too, or a scale artefact reports worse than the start.
            best, best_amp = out["point"]["loss"], point_amp
            while (not es.stop()
                   and calls["n"] // ENSEMBLE_COST + es.popsize <= MAX_EVALS):
                xs = es.ask()
                vals = [ensemble_loss(x) for x in xs]
                es.tell(xs, vals)
                i = int(np.argmin(vals))
                if vals[i] < best:
                    best, best_amp = vals[i], np.clip(xs[i], 0.0, AMP_MAX)
            cma_runs.append({"sigma0": sigma0, "seed": seed, "loss": float(best),
                             "amp": np.asarray(best_amp, float),
                             "evals": calls["n"] // ENSEMBLE_COST,
                             "trace": np.asarray(trace["cur"], float)})
            print(f"  sigma0={sigma0:5.3f} seed={seed}  best {best:.4e}  "
                  f"({calls['n'] // ENSEMBLE_COST} evals)")

    best_cma = min(cma_runs, key=lambda r: r["loss"])
    cma_med = float(np.median([r["loss"] for r in cma_runs]))
    out["cma_es"] = report(
        f"CMA-ES best (s={best_cma['sigma0']}, seed {best_cma['seed']})",
        best_cma["amp"], best_cma["loss"])
    out["cma_es"]["evals"] = best_cma["evals"]

    print(f"\nNelder-Mead, explicit initial simplex at {len(NM_SCALES)} scales:")
    from scipy.optimize import minimize
    nm_runs = []
    for sx in NM_SCALES:
        calls["n"] = 0
        trace["cur"] = []
        simplex = np.vstack(
            [point_amp]
            + [np.clip(point_amp + sx * np.eye(NB)[k], 0.0, AMP_MAX)
               for k in range(NB)]
        )
        r = minimize(ensemble_loss, point_amp, method="Nelder-Mead",
                     options={"initial_simplex": simplex, "maxfev": MAX_EVALS,
                              "xatol": 1e-8, "fatol": 1e-12})
        nm_runs.append({"scale": sx, "loss": float(r.fun),
                        "amp": np.clip(r.x, 0.0, AMP_MAX),
                        "evals": calls["n"] // ENSEMBLE_COST,
                        "trace": np.asarray(trace["cur"], float)})
        print(f"  simplex={sx:5.3f}  best {float(r.fun):.4e}  "
              f"({calls['n'] // ENSEMBLE_COST} evals)")

    best_nm = min(nm_runs, key=lambda r: r["loss"])
    out["nelder_mead"] = report(f"Nelder-Mead best (dx={best_nm['scale']})",
                                best_nm["amp"], best_nm["loss"])
    out["nelder_mead"]["evals"] = best_nm["evals"]

    print("\nagainst E5b's Adam ensemble design:")
    for name, med in (("cma_es", cma_med), ("nelder_mead", None)):
        r = out[name]
        verdict = "worse than" if r["loss"] > adam_loss else "AT OR BETTER THAN"
        print(f"  {name:12} {r['loss'] / adam_loss:8.2f}x Adam's objective "
              f"({verdict} Adam) after {r['evals']} evaluations")
        if med is not None:
            print(f"               median across the grid: {med:.4e} "
                  f"({med / adam_loss:.2f}x)")
        print(f"               held-out {r['correct']}/{r['n']}, "
              f"p5 {r['p5']:+.4f} m against Adam's {adam_p5:+.4f} m")

    assert all(np.isfinite(out[k]["loss"]) for k in out), "non-finite objective"

    np.savez(
        OUT_PATH,
        budget_solves=BUDGET_SOLVES, max_evals=MAX_EVALS,
        budget_solves_eval=BUDGET_SOLVES_EVAL, max_evals_eval=MAX_EVALS_EVAL,
        ensemble_cost=ENSEMBLE_COST, grad_charge=GRAD_CHARGE, adam_iters=N_ITERS,
        cma_median_loss=cma_med,
        cma_grid_sigmas=np.asarray(SIGMAS), cma_grid_seeds=np.asarray(CMA_SEEDS),
        cma_grid_losses=np.asarray([r["loss"] for r in cma_runs]),
        nm_scales=np.asarray(NM_SCALES),
        nm_losses=np.asarray([r["loss"] for r in nm_runs]),
        cma_best_trace=best_cma["trace"], nm_best_trace=best_nm["trace"],
        **{f"{k}_{f}": v[f] for k, v in out.items()
           for f in ("loss", "correct", "n", "p5", "amp", "margins")},
        cma_es_evals=out["cma_es"]["evals"],
        nelder_mead_evals=out["nelder_mead"]["evals"],
    )
    print(f"\nwrote {OUT_PATH.name}")


if __name__ == "__main__":
    main()
