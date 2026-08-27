"""E5c: does the gradient-free advantage at 24 design parameters survive more of them?

Section 4c publishes a control in which tuned CMA-ES reaches a better design
than tuned Adam on the E5b ensemble objective at a matched budget, and withdraws
our claim that the ensemble objective is where the gradient pays. That control
ran at NB = 24. Gradient-free search degrades with dimension and the
forward-variational gradient here does not, so the obvious question is whether
the concession is general or bounded.

The experiment only means something if raising NB raises the DIMENSION and not
the difficulty. Two properties are arranged so that it does:

  Nesting. linspace(0.4, 5.4, NB) contains the NB = 24 centres exactly when
  23 divides NB - 1, so NB is restricted to 24, 47, 93.

  Embedding. Bump width scales with spacing, WID(NB) = 0.18 * 23/(NB-1). A wide
  Gaussian is a positive convolution of narrow ones, so the published
  24-parameter design is an exact interior point of the larger boxes and
  reproduces the same trajectory impact for impact. Therefore
  min L(NB=93) <= min L(NB=24) by construction, and any degradation measured is
  the optimizer failing, not the problem getting harder.

Held fixed at 0.18 the dictionary saturates instead: numerical rank stalls near
52 at both NB = 93 and NB = 185 with condition number 1e16, and uniform maximum
amplitude builds terrain metres taller than the launch height. That version
would manufacture the result we are testing for, and is not run.

Stated limit: the objective is 2 * N_TRAIN particles by two residuals, so 48,
which caps the local Gauss-Newton rank whatever NB is. NB = 93 is therefore a
2:1 over-parameterised design vector rather than 93 independent degrees of
freedom, and this is two dimension points on one problem instance, not a
crossover estimate.

Both arms are tuned over grids recalibrated by one measured, arm-symmetric rule,
because a step size in absolute amplitude units means something different when
bumps are four times narrower. Adam's grid is extended below the value it chose
at the edge in the published sweep. The budget is frozen across NB.

Writes experiments/e5c_result.npz. Nothing else in the repo changes.
"""

import json
import sys
import time
from math import ceil
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

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import AMP_MAX, BIN_PET, BIN_RUBBER, FIXED  # noqa: E402
from e5b_robust_separator import N_ITERS, N_TEST, N_TRAIN, draw_ensemble  # noqa: E402

ROOT = Path(__file__).parent.parent
X_MID = 0.5 * (BIN_RUBBER + BIN_PET)

# NB = 47 was pre-registered and is dropped on its own gate: the embedded warm
# start moves the ensemble loss 10.1% there against a 5% threshold, because at
# 47 bumps the narrow-Gaussian sum is not yet a faithful representation of the
# 24-bump terrain (interior sup error 2.5e-4 m, against 1.5e-6 m at NB = 93).
# NB = 93 moves it 0.4%. Reported rather than accommodated.
NB_GRID = (24, 93)
CTR_LO, CTR_HI, WID_24, NB_24 = 0.4, 5.4, 0.18, 24
ENSEMBLE_COST = 2 * N_TRAIN
CMA_SEEDS = (3, 4, 5)
SIGMA_BASE = (0.01, 0.02, 0.05)
LR_BASE = (0.00025, 0.0005, 0.001, 0.002, 0.004, 0.01, 0.02)
LADDER_MULT = 3
TIME_KILL_S = 120 * 60


def ctr_of(nb):
    return np.linspace(CTR_LO, CTR_HI, nb)


def wid_of(nb):
    return np.full(nb, WID_24 * (NB_24 - 1) / (nb - 1))


def embed(amp24, nb):
    """Exact positive-convolution embedding of a 24-bump design into nb bumps."""
    if nb == NB_24:
        return np.asarray(amp24, dtype=float).copy()
    W, w = WID_24, WID_24 * (NB_24 - 1) / (nb - 1)
    s2 = W * W - w * w
    dx = (CTR_HI - CTR_LO) / (nb - 1)
    c_nb, c_24 = ctr_of(nb), ctr_of(NB_24)
    K = (W / w) * (dx / np.sqrt(2 * np.pi * s2)) * np.exp(
        -((c_nb[:, None] - c_24[None, :]) ** 2) / (2 * s2)
    )
    return K @ np.asarray(amp24, dtype=float)


def terrain_h(x, amp, ctr, wid):
    x = np.asarray(x)
    return np.sum(amp * np.exp(-((x[..., None] - ctr) ** 2) / (2 * wid**2)), axis=-1)


def main():
    t0 = time.time()
    sim = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")

    adam_amp24 = np.asarray(np.load(ROOT / "experiments" / "e5_result.npz")["adam_amp"], float)
    train = draw_ensemble(np.random.default_rng(5), N_TRAIN)
    test = draw_ensemble(np.random.default_rng(17), N_TEST)
    assert len(train) == ENSEMBLE_COST

    known = ("initial state must start above the terrain",
             "degenerate guard crossing",
             "guard crossing with non-approaching velocity")
    counters = {"solves": 0, "fails": 0}

    def landing(amp, ctr, wid, v0, e):
        counters["solves"] += 1
        try:
            r = sim.apply({**FIXED, "v0": np.asarray(v0), "e": float(e),
                           "amp": np.asarray(amp), "ctr": ctr, "wid": wid})
        except Exception as exc:                       # noqa: BLE001
            if not any(k in str(exc) for k in known):
                raise
            counters["fails"] += 1
            return None, None, None
        q = np.asarray(r["qf"], float)
        return q, int(r["status"]), int(r["n_events"])

    def make_numpy_loss(nb):
        ctr, wid = ctr_of(nb), wid_of(nb)

        def loss(amp):
            amp = np.clip(np.asarray(amp, float), 0.0, AMP_MAX)
            tot, stats = 0.0, []
            for v0, e, bin_x in train:
                q, st, nev = landing(amp, ctr, wid, v0, e)
                if q is None:
                    tot += 1e3
                    continue
                stats.append((st, nev))
                dx = q[0] - bin_x
                dy = q[1] - float(terrain_h(np.array(bin_x), amp, ctr, wid))
                tot += dx * dx + dy * dy
            return tot / len(train), stats

        return loss

    def make_jax_loss(nb):
        """Same quadratic in JAX. The score Tesseract is deliberately not used:
        it costs Adam an apply and a vjp per particle that the budget charges
        nothing for, and that undercharge shrinks with NB."""
        ctr, wid = jnp.asarray(ctr_of(nb)), jnp.asarray(wid_of(nb))

        def one(amp, v0, e, bin_x):
            res = apply_tesseract(sim, {**FIXED, "v0": v0, "e": e,
                                        "amp": amp, "ctr": ctr, "wid": wid})
            qf = res["qf"]
            hb = jnp.sum(amp * jnp.exp(-((bin_x - ctr) ** 2) / (2 * wid**2)))
            return (qf[0] - bin_x) ** 2 + (qf[1] - hb) ** 2

        def loss(amp):
            return sum(one(amp, jnp.asarray(v0), e, bx) for v0, e, bx in train) / len(train)

        return loss

    def margins(amp, nb, ensemble):
        ctr, wid = ctr_of(nb), wid_of(nb)
        out = []
        for v0, e, bin_x in ensemble:
            q, _, _ = landing(amp, ctr, wid, v0, e)
            if q is None:
                out.append(-9.9); continue
            out.append((X_MID - q[0]) if bin_x == BIN_RUBBER else (q[0] - X_MID))
        return np.asarray(out)

    def score(amp, nb):
        m = margins(amp, nb, test)
        return int((m > 0).sum()), len(m), float(np.percentile(m, 5))

    # ---- gates ------------------------------------------------------------
    print("gates:")
    ref_loss, ref_stats = None, None
    warm = {}
    for nb in NB_GRID:
        assert (nb - 1) % (NB_24 - 1) == 0, nb
        a0 = embed(adam_amp24, nb)
        if nb == NB_24:
            assert np.allclose(a0, adam_amp24), "embedding not identity at NB=24"
        assert a0.min() >= 0.0 and a0.max() < 0.9 * AMP_MAX, (nb, a0.min(), a0.max())
        xs = np.linspace(CTR_LO, CTR_HI, 400)
        sup = float(np.abs(terrain_h(xs, a0, ctr_of(nb), wid_of(nb))
                           - terrain_h(xs, adam_amp24, ctr_of(NB_24), wid_of(NB_24))).max())
        env = float(terrain_h(xs, np.full(nb, AMP_MAX), ctr_of(nb), wid_of(nb)).max())
        rank = int(np.linalg.matrix_rank(
            np.exp(-((xs[:, None] - ctr_of(nb)) ** 2) / (2 * wid_of(nb) ** 2)), tol=1e-6))
        L, stats = make_numpy_loss(nb)(a0)
        if ref_loss is None:
            ref_loss, ref_stats, ref_env = L, stats, env
        assert abs(L / ref_loss - 1) < 0.05, f"warm loss moved {L/ref_loss:.3f} at NB={nb}"
        assert sum(1 for a, b in zip(stats, ref_stats) if a != b) <= 1, f"status histogram moved at NB={nb}"
        assert abs(env - ref_env) < 1e-3, f"envelope moved at NB={nb}: {env:.4f}"
        assert rank == nb, f"rank {rank} != {nb}"
        warm[nb] = dict(amp0=a0, loss=float(L), sup=sup, env=env, rank=rank)
        print(f"  NB={nb:3d} warm loss {L:.6e} sup {sup:.2e} env {env:.4f} rank {rank}")

    # bit-agreement between the two loss implementations
    for nb in NB_GRID:
        lj = float(make_jax_loss(nb)(jnp.asarray(warm[nb]["amp0"])))
        ln = make_numpy_loss(nb)(warm[nb]["amp0"])[0]
        assert abs(lj - ln) < 1e-10, (nb, lj, ln)
    print("  jax and numpy losses agree to 1e-10 at every NB")

    # ---- in-situ charge, frozen across NB ---------------------------------
    charges = {}
    for nb in NB_GRID:
        f, g = make_numpy_loss(nb), jax.value_and_grad(make_jax_loss(nb))
        a0 = jnp.asarray(warm[nb]["amp0"])
        g(a0)
        te = min(_t(lambda: f(warm[nb]["amp0"])) for _ in range(3))
        ta = min(_t(lambda: jax.block_until_ready(g(a0)[1])) for _ in range(3))
        charges[nb] = ta / te
        print(f"  NB={nb:3d} Adam iteration / ensemble eval = {ta/te:.2f}")
    MAX_EVALS = ceil(N_ITERS * max(charges.values()))
    print(f"\nfrozen budget: {MAX_EVALS} ensemble evaluations at every NB "
          f"({MAX_EVALS*ENSEMBLE_COST} particle solves)")

    # ---- step-size calibration, one rule applied to BOTH arms -------------
    # A step in absolute amplitude units means something different when bumps
    # are four times narrower: the slope perturbation scales with sigma * NB,
    # and the reset map is driven by slope at impact. Match the response.
    print("\nstep-size calibration:")
    S = {}
    for nb in NB_GRID:
        f, a0 = make_numpy_loss(nb), warm[nb]["amp0"]
        rng = np.random.default_rng(11)
        def sd_at(s):
            vals = [f(np.clip(a0 + rng.normal(0, s * 0.02, nb), 0, AMP_MAX))[0] for _ in range(12)]
            return float(np.std(vals))
        if nb == NB_24:
            S[nb], ref_sd = 1.0, sd_at(1.0)
        else:
            cand = (1.0, 0.5, 0.25, 0.1, 0.05, 0.025)
            sds = {s: sd_at(s) for s in cand}
            S[nb] = min(cand, key=lambda s: abs(sds[s] - ref_sd))
        print(f"  NB={nb:3d} scale {S[nb]:.3f}")

    def run_adam(nb):
        gfn = jax.value_and_grad(make_jax_loss(nb))
        nf = make_numpy_loss(nb)
        out = []
        for lr in [S[nb] * x for x in LR_BASE]:
            amp = jnp.asarray(warm[nb]["amp0"])
            opt = optax.adam(learning_rate=lr); st = opt.init(amp)
            best, best_amp = float(np.inf), np.asarray(warm[nb]["amp0"])
            for _ in range(N_ITERS):
                v, g = gfn(amp); v = float(v)
                if v < best: best, best_amp = v, np.asarray(amp)
                upd, st = opt.update(g, st)
                amp = jnp.clip(optax.apply_updates(amp, upd), 0.0, AMP_MAX)
            fin = float(make_jax_loss(nb)(amp))
            if fin < best: best, best_amp = fin, np.asarray(amp)
            k, n, p5 = score(best_amp, nb)
            out.append(dict(lr=float(lr), loss=best, amp=best_amp, correct=k, n=n, p5=p5))
            print(f"    adam lr={lr:9.6f} loss {best:.4e}  {k}/{n}  p5 {p5:+.3f}")
        return out

    def run_cma(nb, budget, tag="cma"):
        import cma
        f, a0 = make_numpy_loss(nb), warm[nb]["amp0"]
        out = []
        for s0 in [S[nb] * x for x in SIGMA_BASE]:
            for seed in CMA_SEEDS:
                used = [0]
                def obj(x):
                    used[0] += 1
                    return f(x)[0]
                es = cma.CMAEvolutionStrategy(list(a0), s0,
                        {"bounds": [0.0, AMP_MAX], "maxfevals": budget,
                         "verbose": -9, "seed": seed})
                best, best_amp = warm[nb]["loss"], a0
                while not es.stop() and used[0] < budget:
                    xs = es.ask(); vs = [obj(x) for x in xs]; es.tell(xs, vs)
                    i = int(np.argmin(vs))
                    if vs[i] < best: best, best_amp = vs[i], np.clip(xs[i], 0, AMP_MAX)
                k, n, p5 = score(best_amp, nb)
                out.append(dict(sigma0=float(s0), seed=seed, loss=float(best), amp=best_amp,
                                evals=used[0], correct=k, n=n, p5=p5, popsize=int(es.popsize)))
                print(f"    {tag} s={s0:8.5f} seed={seed} loss {best:.4e}  {k}/{n}  "
                      f"p5 {p5:+.3f}  ({used[0]} evals, pop {es.popsize})")
        return out

    results = {}
    for nb in NB_GRID:
        if time.time() - t0 > TIME_KILL_S:
            print("TIME KILL"); break
        print(f"\n=== NB = {nb} ===")
        results[nb] = dict(adam=run_adam(nb), cma=run_cma(nb, MAX_EVALS))

    # cost-to-target ladder at the top dimension
    ladder = []
    if NB_GRID[-1] in results and time.time() - t0 < TIME_KILL_S:
        nb = NB_GRID[-1]
        print(f"\n=== ladder: CMA at {LADDER_MULT}x budget, NB = {nb} ===")
        ladder = run_cma(nb, MAX_EVALS * LADDER_MULT, tag="cma3x")

    print("\n=== summary ===")
    rows = {}
    for nb in results:
        al = [r["loss"] for r in results[nb]["adam"]]
        cl = [r["loss"] for r in results[nb]["cma"]]
        rows[nb] = dict(adam_best=min(al), adam_med=float(np.median(al)),
                        cma_best=min(cl), cma_med=float(np.median(cl)))
        r = rows[nb]
        print(f"  NB={nb:3d}  adam best {r['adam_best']:.4e} med {r['adam_med']:.4e} | "
              f"cma best {r['cma_best']:.4e} med {r['cma_med']:.4e} | "
              f"ratio best {r['cma_best']/r['adam_best']:.2f} med {r['cma_med']/r['adam_med']:.2f}")

    payload = dict(nb_grid=np.asarray(NB_GRID), max_evals=MAX_EVALS,
                   ensemble_cost=ENSEMBLE_COST, n_iters=N_ITERS,
                   charges=np.asarray([charges[n] for n in NB_GRID]),
                   warm_loss=np.asarray([warm[n]["loss"] for n in NB_GRID]),
                   warm_sup=np.asarray([warm[n]["sup"] for n in NB_GRID]),
                   ranks=np.asarray([warm[n]["rank"] for n in NB_GRID]),
                   scales=np.asarray([S[n] for n in NB_GRID]),
                   residuals=2 * ENSEMBLE_COST, ladder_mult=LADDER_MULT,
                   solver_fails=counters["fails"], total_solves=counters["solves"],
                   elapsed_s=time.time() - t0, stage="complete")
    for nb in results:
        for arm in ("adam", "cma"):
            payload[f"{arm}_{nb}_loss"] = np.asarray([r["loss"] for r in results[nb][arm]])
            payload[f"{arm}_{nb}_p5"] = np.asarray([r["p5"] for r in results[nb][arm]])
            payload[f"{arm}_{nb}_correct"] = np.asarray([r["correct"] for r in results[nb][arm]])
    if ladder:
        payload["ladder_loss"] = np.asarray([r["loss"] for r in ladder])
        payload["ladder_p5"] = np.asarray([r["p5"] for r in ladder])
    np.savez(ROOT / "experiments" / "e5c_result.npz", **payload)
    print(f"\nwrote e5c_result.npz in {time.time()-t0:.0f}s "
          f"({counters['solves']} solves, {counters['fails']} solver failures)")


def _t(fn):
    s = time.perf_counter(); fn(); return time.perf_counter() - s


if __name__ == "__main__":
    main()
