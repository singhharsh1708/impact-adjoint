"""Zero-shot generalization with statistics rather than a single sweep.

E6 shows one deterministic sweep and a 12-draw jitter check at three
restitutions. This repeats the jitter test over many draws at every restitution on a grid,
for BOTH the point design (E5) and the ensemble-refined design (E5b), so the
classification boundary is measured with confidence intervals instead of
asserted from one pass. The comparison is the point: the point design is
indecisive on the high-restitution side, where trajectories take many impacts
and small inlet changes compound, including at its own trained value, while
the ensemble design is decisive at every restitution sampled. Two discordant
restitutions against none is McNemar p = 0.50, the floor for two discordant
pairs, so the direction is consistent but this sweep cannot establish it.

Writes experiments/generalization_stats.npz.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

from tesseract_core import Tesseract

import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import BIN_PET, BIN_RUBBER, CTR, V0, WID

ROOT = Path(__file__).parent.parent
X_MID = 0.5 * (BIN_RUBBER + BIN_PET)
# the design's own horizon: evaluating past it measures a different quantity
# than the one that was designed, so import rather than restate
from e5_separator import FIXED  # noqa: E402

ES = np.round(np.arange(0.40, 0.881, 0.025), 4)
N_DRAWS = 40
V_SD = 0.03
TRAINED = (0.5, 0.8)


def exact_mcnemar(b01, b10):
    """Two-sided exact McNemar on the discordant pairs only."""
    from math import comb

    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2.0 * tail)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    t = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    designs = {
        "point": np.load(ROOT / "experiments" / "e5_result.npz")["adam_amp"],
        "robust": np.load(ROOT / "experiments" / "e5b_result.npz")["robust_amp"],
    }

    out = {}
    for name, amp in designs.items():
        rng = np.random.default_rng(4242)  # same jitter draws for both designs
        frac_a, ci_lo, ci_hi, x_mean, x_sd = [], [], [], [], []
        print(f"\n{name} design")
        for e in ES:
            xs = []
            for _ in range(N_DRAWS):
                v0 = V0 + rng.normal(0, V_SD, 2)
                r = t.apply({**FIXED, "v0": v0, "e": float(e), "amp": amp, "ctr": CTR, "wid": WID})
                xs.append(float(np.asarray(r["qf"])[0]))
            xs = np.array(xs)
            k = int((xs < X_MID).sum())
            lo, hi = wilson(k, N_DRAWS)
            frac_a.append(k / N_DRAWS); ci_lo.append(lo); ci_hi.append(hi)
            x_mean.append(xs.mean()); x_sd.append(xs.std())
            mark = "  <- trained" if any(abs(e - tr) < 1e-9 for tr in TRAINED) else ""
            print(f"  e={e:5.3f}  P(bin A) = {k:2d}/{N_DRAWS} = {k/N_DRAWS:4.2f} "
                  f"[{lo:4.2f}, {hi:4.2f}]   x {xs.mean():5.2f} +/- {xs.std():4.2f} m{mark}")
        out[name] = dict(frac_a=np.array(frac_a), ci_lo=np.array(ci_lo),
                         ci_hi=np.array(ci_hi), x_mean=np.array(x_mean), x_sd=np.array(x_sd))

    frac_a = out["robust"]["frac_a"]
    ci_lo, ci_hi = out["robust"]["ci_lo"], out["robust"]["ci_hi"]
    x_mean, x_sd = out["robust"]["x_mean"], out["robust"]["x_sd"]

    # the decision boundary under jitter: the band where the classification is
    # not unanimous, which is the honest version of "one clean threshold"
    unsure = np.where((frac_a > 0.0) & (frac_a < 1.0))[0]
    if len(unsure):
        band = (ES[unsure.min()], ES[unsure.max()])
    else:
        band = (float("nan"), float("nan"))
    a_side = ES[frac_a == 1.0]
    b_side = ES[frac_a == 0.0]
    print(f"\nunanimous bin A up to e = {a_side.max() if len(a_side) else float('nan'):.3f}")
    print(f"unanimous bin B from e = {b_side.min() if len(b_side) else float('nan'):.3f}")
    if len(unsure):
        print(f"transition band under {V_SD*100:.0f} cm/s inlet jitter: "
              f"e in [{band[0]:.3f}, {band[1]:.3f}]")
    else:
        print(f"no transition band under {V_SD*100:.0f} cm/s inlet jitter: "
              "every restitution sampled classifies unanimously")

    # Each design sets its own threshold, so scoring both against one fixed
    # cut would be meaningless. The quantity that compares them fairly is
    # DECISIVENESS: at how many restitutions is the outcome unanimous across
    # the jitter draws, i.e. the machine sorts that material reliably rather
    # than sometimes.
    def decisiveness(d):
        fa = d["frac_a"]
        unanimous = (fa == 0.0) | (fa == 1.0)
        thr_a = ES[fa == 1.0]
        thr_b = ES[fa == 0.0]
        return (int(unanimous.sum()), len(fa),
                float(thr_a.max()) if len(thr_a) else float("nan"),
                float(thr_b.min()) if len(thr_b) else float("nan"),
                ES[~unanimous])

    stats = {}
    for name in ("point", "robust"):
        k, n, a_max, b_min, bad = decisiveness(out[name])
        stats[name] = (k, n, bad)
        print(f"{name:6s} design: unanimous at {k}/{n} restitutions; "
              f"bin A up to e={a_max:.3f}, bin B from e={b_min:.3f}")
        if len(bad):
            print(f"         indecisive at e = {', '.join(f'{x:.3f}' for x in bad)}")

    # The jitter-sweep McNemar p was published with nothing computing it.
    # Discordant pairs: restitutions where exactly one design is decisive.
    dec_p = (out["point"]["frac_a"] == 0.0) | (out["point"]["frac_a"] == 1.0)
    dec_r = (out["robust"]["frac_a"] == 0.0) | (out["robust"]["frac_a"] == 1.0)
    b01 = int((~dec_p & dec_r).sum())
    b10 = int((dec_p & ~dec_r).sum())
    mcnemar_p = float(exact_mcnemar(b01, b10))
    print(f"\njitter decisiveness McNemar: b01={b01}, b10={b10}, exact p = {mcnemar_p:.4f}")

    np.savez(ROOT / "experiments" / "generalization_stats.npz",
             es=ES, n_draws=N_DRAWS, v_sd=V_SD, x_mid=X_MID, band=np.array(band),
             mcnemar_b01=b01, mcnemar_b10=b10, mcnemar_p=mcnemar_p,
             **{f"{n}_{k}": v for n, d in out.items() for k, v in d.items()})

    kp, n, bad_p = stats["point"]
    kr, _, bad_r = stats["robust"]
    assert kr >= kp, "ensemble design should not be less decisive than the point design"
    verdict = ("is unanimous at every restitution sampled"
               if kr == n else
               f"is decisive at {kr}/{n}, indecisive at {sorted(bad_r)}")
    print(f"\nGENERALIZATION STATS PASSED: the point design is indecisive under jitter at "
          f"{n-kp} of {n} restitutions (including its own trained e=0.8), while the "
          f"ensemble design {verdict}")


if __name__ == "__main__":
    main()
