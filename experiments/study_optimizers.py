"""Multi-seed optimizer benchmark on the 24-dimensional separator design.

The single-seed comparison in e5_separator.py is not enough to claim anything
about method performance. This runs every method from several random starts,
records the full trace, and reports medians with interquartile bands under
BOTH cost accountings:

  eval-count : a gradient call charged as 2 forward solves (generous to us)
  wall-clock : a gradient call charged at its measured cost, remeasured here
               as the wall-clock cost of the endpoint as it is actually
               called (it builds every sensitivity column whatever the
               cotangent asks for)

Both methods get the same tuning treatment: CMA-ES over a sigma0 grid and
Adam over a learning-rate grid, best of each per seed. Tuning only the
baseline, or only ours, would decide the outcome by the protocol rather than
by the method.

Writes experiments/optimizer_benchmark.npz.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

import cma
import jax
import jax.numpy as jnp
import optax
from scipy.optimize import minimize
from tesseract_core import Tesseract

jax.config.update("jax_enable_x64", True)

import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import AMP_MAX, BUDGET, NB, make_objective

ROOT = Path(__file__).parent.parent

SEEDS = [0, 1, 2, 3, 4]
# BUDGET comes from e5_separator so the benchmark cannot silently run a
# different budget than the design it benchmarks.
GRAD_CHARGE_EVAL = 2  # units per gradient call, eval-count accounting


def measure_grad_charge(sim):
    """Wall-clock cost of one gradient, in forward solves, for THIS objective.

    Previously hardcoded at 8.5 from study_scaling.py, and previously explained
    here as being measured "against the 24 columns this objective actually
    differentiates". That explanation was wrong: vector_jacobian_product calls
    _run(want_sens=True) and slices afterwards, so it builds every sensitivity
    column whatever the cotangent asks for. What this measures is the wall-clock
    cost of the endpoint as it is actually called, which is the number the
    budget accounting needs; it was never a per-column figure.
    """
    import time

    from e5_separator import CTR, FIXED, V0, WID

    amp = np.full(NB, 0.05)
    cfg = {**FIXED, "v0": V0, "e": 0.5, "amp": amp, "ctr": CTR, "wid": WID}
    cot = {"qf": np.array([1.0, 0.0, 0.0, 0.0])}

    def timeit(fn, n=12):
        fn()
        return min(_t for _ in range(n)
                   for _t in [(lambda s=time.perf_counter(): (fn(), time.perf_counter() - s)[1])()])

    t_apply = timeit(lambda: sim.apply(cfg))
    t_vjp = timeit(lambda: sim.vector_jacobian_product(
        cfg, vjp_inputs={"amp"}, vjp_outputs={"qf"}, cotangent_vector=cot))
    charge = t_vjp / t_apply
    print(f"measured gradient charge: {t_vjp * 1000:.2f} ms / {t_apply * 1000:.2f} ms = "
          f"{charge:.2f} forward solves")
    return charge
SIGMAS = [0.02, 0.05, 0.1]
LRS = [0.01, 0.02, 0.05]


def start_point(seed):
    rng = np.random.default_rng(1000 + seed)
    return np.clip(rng.uniform(0.0, 0.12, NB), 0.0, AMP_MAX)


def run_adam(sim, score, x0, charge, lr):
    grad_fn = jax.value_and_grad(lambda a: _loss_j(sim, score, a))
    amp = jnp.asarray(x0)
    opt = optax.adam(learning_rate=lr)
    st = opt.init(amp)
    used, best, trace = 0.0, np.inf, []
    step_cost = 2 + 2 * charge  # 2 apply + 2 gradient calls per iteration
    while used + step_cost <= BUDGET:
        val, g = grad_fn(amp)
        used += step_cost
        best = min(best, float(val))
        trace.append((used, best))
        upd, st = opt.update(g, st)
        amp = jnp.clip(optax.apply_updates(amp, upd), 0.0, AMP_MAX)
    return np.array(trace)


def _loss_j(sim, score, amp):
    from e5_separator import BIN_PET, BIN_RUBBER, CTR, E_PET, E_RUBBER, FIXED, V0, WID
    from tesseract_jax import apply_tesseract

    def one(e, bin_x):
        res = apply_tesseract(sim, {**FIXED, "v0": V0, "e": e, "amp": amp, "ctr": CTR, "wid": WID})
        tgt = jnp.stack([jnp.asarray(bin_x), jnp.sum(amp * jnp.exp(-((bin_x - CTR) ** 2) / (2 * WID**2)))])
        return apply_tesseract(score, {"qf": res["qf"], "target": tgt,
                                       "weights": jnp.array([1.0, 1.0, 0.0])})["loss"]

    return one(E_RUBBER, BIN_RUBBER) + one(E_PET, BIN_PET)


def run_nm(sim, score, x0):
    loss_np, _, _ = make_objective(sim, score)
    used, best, trace = [0.0], [np.inf], []

    def f(v):
        used[0] += 2
        val = loss_np(v)
        best[0] = min(best[0], val)
        trace.append((used[0], best[0]))
        return val

    minimize(f, x0.copy(), method="Nelder-Mead",
             options={"maxfev": BUDGET // 2, "xatol": 1e-8, "fatol": 1e-10})
    return np.array(trace)


def run_cma(sim, score, x0, sigma0, seed):
    loss_np, _, _ = make_objective(sim, score)
    used, best, trace = 0.0, np.inf, []
    es = cma.CMAEvolutionStrategy(x0.copy(), sigma0,
                                  {"bounds": [0.0, AMP_MAX], "verbose": -9, "seed": seed + 1})
    while used + 2 * es.popsize <= BUDGET:
        xs = es.ask()
        vals = []
        for x in xs:
            used += 2
            v = loss_np(x)
            best = min(best, v)
            trace.append((used, best))
            vals.append(v)
        es.tell(xs, vals)
    return np.array(trace)


def resample(trace, grid):
    """Best-so-far value at each budget point on a common grid."""
    out = np.full(len(grid), np.nan)
    if len(trace) == 0:
        return out
    u, b = trace[:, 0], trace[:, 1]
    for i, g in enumerate(grid):
        m = u <= g
        out[i] = b[m][-1] if m.any() else np.nan
    return out


def main():
    sim = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    score = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "score_target" / "tesseract_api.py")
    grid = np.linspace(20, BUDGET, 60)
    grad_charge_wall = measure_grad_charge(sim)

    out = {}
    for charge, tag in ((GRAD_CHARGE_EVAL, "eval"), (grad_charge_wall, "wall")):
        curves = []
        for s in SEEDS:
            best_c, best_f, best_lr = None, np.inf, None
            for lr in LRS:
                tr = run_adam(sim, score, start_point(s), charge, lr)
                if tr[-1, 1] < best_f:
                    best_f, best_c, best_lr = tr[-1, 1], resample(tr, grid), lr
            curves.append(best_c)
            print(f"adam[{tag}] seed {s} (best over lr {LRS}): final {best_f:.3e} at lr={best_lr}")
        out[f"adam_{tag}"] = np.array(curves)

    nm = []
    for s in SEEDS:
        tr = run_nm(sim, score, start_point(s))
        nm.append(resample(tr, grid))
        print(f"nelder-mead seed {s}: final {tr[-1,1]:.3e}")
    out["nm"] = np.array(nm)

    cma_curves = []
    for s in SEEDS:
        best_curve, best_final = None, np.inf
        for sg in SIGMAS:
            tr = run_cma(sim, score, start_point(s), sg, s)
            if tr[-1, 1] < best_final:
                best_final, best_curve = tr[-1, 1], resample(tr, grid)
        cma_curves.append(best_curve)
        print(f"cma-es seed {s} (best over sigma0 {SIGMAS}): final {best_final:.3e}")
    out["cma"] = np.array(cma_curves)

    np.savez(ROOT / "experiments" / "optimizer_benchmark.npz", grid=grid,
             seeds=np.array(SEEDS), grad_charge_wall=grad_charge_wall, **out)

    def med(k):
        return np.nanmedian(out[k][:, -1])

    print(f"\nfinal medians over {len(SEEDS)} seeds")
    print(f"  adam (eval accounting) : {med('adam_eval'):.3e}")
    print(f"  adam (wall accounting) : {med('adam_wall'):.3e}")
    print(f"  cma-es (tuned sigma0)  : {med('cma'):.3e}")
    print(f"  nelder-mead            : {med('nm'):.3e}")

    ratio_eval = med("cma") / med("adam_eval")
    ratio_wall = med("cma") / med("adam_wall")
    print(f"\nCMA / Adam ratio: {ratio_eval:.2f}x under eval accounting, "
          f"{ratio_wall:.2f}x under wall-clock accounting")
    print("(above 1 means Adam is ahead; both methods tuned over a grid per seed)")
    print("\nOPTIMIZER BENCHMARK PASSED: multi-seed medians recorded under both "
          "accountings, with both methods tuned")


if __name__ == "__main__":
    main()
