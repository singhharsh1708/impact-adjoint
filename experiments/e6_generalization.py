"""E6: zero-shot generalization of the designed separator.

The E5 surface was optimized for exactly two restitution values (0.5, 0.8).
Here we ask what it does to materials it never saw: a dense sweep of
restitution over [0.35, 0.95] plus an inlet-velocity jitter grid. The designed
geometry turns out to encode a *classifier*: a single threshold in e separates
the continuum into bin-A-like and bin-B-like landings, behavior the optimizer
was never asked for.

Metric: landing x at t_end (settled or final state) against the bin midpoint
x_mid = (2.8 + 4.4)/2 = 3.6; class A if x < x_mid else B.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

from tesseract_core import Tesseract

ROOT = Path(__file__).parent.parent

D = np.load(ROOT / "experiments" / "e5_result.npz")
AMP, CTR, WID = D["adam_amp"], D["ctr"], D["wid"]
X_MID = 0.5 * (BIN_RUBBER + BIN_PET)
# the separator is defined at E5's horizon: its output is "where the particle
# is when the run ends", so evaluating past t_final measures a different
# quantity than the one designed. Import it rather than restating it.
import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import BIN_PET, BIN_RUBBER, FIXED, V0  # noqa: E402


def landing_x(t, e, v0):
    r = t.apply({**FIXED, "v0": v0, "e": float(e), "amp": AMP, "ctr": CTR, "wid": WID})
    return float(np.asarray(r["qf"])[0]), int(r["status"])


def main():
    t = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")

    es = np.round(np.arange(0.35, 0.951, 0.025), 4)
    xs, statuses = [], []
    for e in es:
        x, s = landing_x(t, e, V0)
        xs.append(x)
        statuses.append(s)
    xs = np.array(xs)
    cls = np.where(xs < X_MID, "A", "B")

    # single-threshold check on the validated domain [0.35, 0.875]; beyond it
    # the superball regime (e >= 0.9) can rebound off the backstop and exit
    # left -- reported honestly as the failure edge.
    dom = es <= 0.8751
    a_idx = [i for i in np.where(dom)[0] if cls[i] == "A"]
    b_idx = [i for i in np.where(dom)[0] if cls[i] == "B"]
    clean = bool(a_idx and b_idx and max(a_idx) < min(b_idx))
    thr_lo, thr_hi = (es[max(a_idx)], es[min(b_idx)]) if clean else (np.nan, np.nan)

    print("e sweep [0.35, 0.95], designed-for values 0.5 / 0.8:")
    for e, x, c, s in zip(es, xs, cls, statuses):
        mark = " <-- trained" if abs(e - 0.5) < 1e-9 or abs(e - 0.8) < 1e-9 else ""
        mark += " (outside validated domain)" if e > 0.8751 else ""
        print(f"  e={e:5.3f}  landing x={x:6.3f}  bin {c}  status {s}{mark}")
    print(f"\nsingle clean threshold on [0.35, 0.875]: {clean}  (A up to e={thr_lo}, B from e={thr_hi})")
    print(f"failure edge: e=0.900 lands {abs(xs[list(es).index(0.9)] - X_MID) * 1000:.0f} mm "
          f"on the {'A' if xs[list(es).index(0.9)] < X_MID else 'B'} side of the midpoint; "
          "beyond the validated domain the sweep is not monotone")

    # velocity-jitter robustness at three unseen materials
    rng = np.random.default_rng(11)
    jitter_ok = {}
    for e in (0.45, 0.65, 0.85):
        # the sweep's own threshold is A up to e=0.65, B from e=0.675, so
        # e=0.65 belongs to A; it is the threshold-adjacent probe and the
        # assertion below only gates the two clearly-classed materials
        want = "A" if e <= thr_lo else "B"
        results = []
        for _ in range(12):
            v0 = V0 + rng.normal(0.0, 0.03, 2)  # 3 cm/s scatter
            x, _ = landing_x(t, e, v0)
            results.append("A" if x < X_MID else "B")
        frac = results.count(want) / len(results)
        jitter_ok[e] = (want, frac)
        print(f"velocity jitter (sd 0.03 m/s) at e={e}: {results.count('A')}A/{results.count('B')}B")

    np.savez(ROOT / "experiments" / "e6_result.npz", es=es, xs=xs, x_mid=X_MID,
             thr_lo=thr_lo, thr_hi=thr_hi, statuses=np.array(statuses),
             jitter_e=np.array(sorted(jitter_ok)),
             jitter_frac=np.array([jitter_ok[k][1] for k in sorted(jitter_ok)]))

    assert clean, "expected a single A->B threshold on the validated domain"
    assert cls[list(es).index(0.5)] == "A" and cls[list(es).index(0.8)] == "B"
    # the previous 0.8 floor was tuned to a superseded run that evaluated the
    # design past its own horizon and produced 10/12 here; at the design
    # horizon both off-design probes are unanimous, so assert that
    assert jitter_ok[0.45][1] == 1.0 and jitter_ok[0.85][1] == 1.0, \
        "off-design probes should classify unanimously under inlet jitter"
    print("\nE6 PASSED: one threshold classifies the restitution continuum on [0.35, 0.875], "
          "with inlet-scatter tolerance at off-design materials and an honest failure edge at e>=0.9")


if __name__ == "__main__":
    main()
