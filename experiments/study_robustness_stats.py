"""Statistics for the separator designs: purity with confidence intervals,
and the separation margin, over repeated independent test ensembles.

The single 200-particle purity number in E5b is one draw and a one-particle
difference. This measures both designs on several independent ensembles,
reports Wilson score intervals rather than bare fractions, and reports the
separation margin, which is the quantity a plant engineer would actually
tune (how far the two landing distributions sit from the decision boundary).

Writes experiments/robustness_stats.npz.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

from tesseract_core import Tesseract

import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import BIN_PET, BIN_RUBBER, CTR, E_PET, E_RUBBER, FIXED, V0, WID

ROOT = Path(__file__).parent.parent
X_MID = 0.5 * (BIN_RUBBER + BIN_PET)
V_SD, E_SD = 0.05, 0.03
N_PER_MAT = 100
ENSEMBLES = 5


def wilson(k, n, z=1.96):
    """Wilson score interval. Correct at the boundary, unlike the normal
    approximation, which gives a zero-width interval at k = n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def draw(rng, n):
    out = []
    for e_mat, bin_x in ((E_RUBBER, BIN_RUBBER), (E_PET, BIN_PET)):
        for _ in range(n):
            out.append((V0 + rng.normal(0, V_SD, 2),
                        float(np.clip(rng.normal(e_mat, E_SD), 0.05, 0.93)), bin_x))
    return out


def evaluate(t, amp, ens):
    xs, want_a = [], []
    for v0, e, bx in ens:
        r = t.apply({**FIXED, "v0": np.asarray(v0), "e": e,
                     "amp": np.asarray(amp), "ctr": CTR, "wid": WID})
        xs.append(float(np.asarray(r["qf"])[0]))
        want_a.append(bx == BIN_RUBBER)
    xs, want_a = np.array(xs), np.array(want_a)
    correct = (xs < X_MID) == want_a
    # margin: signed distance to the boundary, positive when on the right side
    margin = np.where(want_a, X_MID - xs, xs - X_MID)
    return correct, margin


def main():
    t = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    d = np.load(ROOT / "experiments" / "e5b_result.npz")
    designs = {"point": d["point_amp"], "robust": d["robust_amp"]}

    res = {}
    for name, amp in designs.items():
        ks, ns, margins = [], [], []
        for i in range(ENSEMBLES):
            ens = draw(np.random.default_rng(500 + i), N_PER_MAT)
            correct, margin = evaluate(t, amp, ens)
            ks.append(int(correct.sum())); ns.append(len(correct))
            margins.append(margin)
        K, N = int(np.sum(ks)), int(np.sum(ns))
        lo, hi = wilson(K, N)
        m = np.concatenate(margins)
        res[name] = dict(k=K, n=N, lo=lo, hi=hi, margins=m, per_ens=np.array(ks))
        print(f"{name:7s}: {K}/{N} correct  ({100*K/N:.2f}%, Wilson 95% CI "
              f"[{100*lo:.2f}, {100*hi:.2f}])   margin median {np.median(m):.3f} m, "
              f"5th pct {np.percentile(m, 5):.3f} m, min {m.min():.3f} m")

    p, r = res["point"], res["robust"]
    # The claim is about the LOW TAIL, not the centre: refinement trades a
    # little median margin for a much safer worst case. Testing stochastic
    # dominance would be the wrong test and would (correctly) fail, so
    # bootstrap the statistic actually claimed.
    # The same particles are evaluated under both designs, so the bootstrap
    # must resample particle INDICES and carry both designs' values together.
    # Resampling the two margin arrays independently discards the pairing.
    rng = np.random.default_rng(0)
    n = len(p["margins"])
    assert len(r["margins"]) == n, "designs must be scored on the same particles"
    diffs = []
    for _ in range(4000):
        idx = rng.integers(0, n, size=n)
        diffs.append(np.percentile(r["margins"][idx], 5)
                     - np.percentile(p["margins"][idx], 5))
    diffs = np.array(diffs)
    lo_d, hi_d = np.percentile(diffs, [2.5, 97.5])
    print(f"\nmedian margin {np.median(p['margins']):.3f} m -> {np.median(r['margins']):.3f} m "
          f"(slightly lower: the centre is traded away)")
    print(f"5th percentile {np.percentile(p['margins'],5):.3f} m -> "
          f"{np.percentile(r['margins'],5):.3f} m")
    # Purity is a paired binary outcome on the same particles, so the
    # criterion is McNemar rather than two independent Wilson intervals.
    from math import comb

    pc, rc = p["margins"] > 0, r["margins"] > 0
    b01 = int((~pc & rc).sum())   # point wrong, robust right
    b10 = int((pc & ~rc).sum())   # point right, robust wrong
    nd = b01 + b10
    if nd:
        k = min(b01, b10)
        pval = min(1.0, 2 * sum(comb(nd, i) for i in range(k + 1)) / 2 ** nd)
    else:
        pval = 1.0
    print(f"McNemar on paired purity: {b01} point-wrong/robust-right, "
          f"{b10} the other way, exact two-sided p = {pval:.3g}")

    print(f"bootstrap 95% CI on the 5th-percentile improvement: "
          f"[{lo_d:+.3f}, {hi_d:+.3f}] m over 4000 resamples")
    print(f"worst case {p['margins'].min():+.3f} m -> {r['margins'].min():+.3f} m")

    np.savez(ROOT / "experiments" / "robustness_stats.npz",
             point_margins=p["margins"], robust_margins=r["margins"],
             point_k=p["k"], point_n=p["n"], robust_k=r["k"], robust_n=r["n"],
             point_ci=np.array([p["lo"], p["hi"]]), robust_ci=np.array([r["lo"], r["hi"]]),
             point_per_ens=p["per_ens"], robust_per_ens=r["per_ens"],
             tail_ci=np.array([lo_d, hi_d]),
             mcnemar_b01=b01, mcnemar_b10=b10, mcnemar_p=pval, n_ensembles=ENSEMBLES, n_per_mat=N_PER_MAT)

    assert r["k"] >= p["k"], "robust design should not classify worse overall"
    assert lo_d > 0, "5th-percentile margin improvement should exclude zero"
    print("\nROBUSTNESS STATS PASSED: purity reported with Wilson intervals over "
          f"{ENSEMBLES} independent ensembles; the low tail of the margin improves "
          "with a bootstrap interval excluding zero")


if __name__ == "__main__":
    main()
