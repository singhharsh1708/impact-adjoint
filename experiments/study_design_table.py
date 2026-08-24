"""Per-design purity and margin intervals for the studies page.

The four rows on the studies page were published with nothing computing them:
the Wilson purity intervals and the bootstrap intervals on the fifth-percentile
margin existed only as typed text. They are generated here, into an artifact
the drift suite compares the page against.

All four designs are scored on the same 200 held-out particles, so the columns
are paired.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
BOOT = 10_000
SEED = 11

DESIGNS = [
    ("Adam", "margins_adam"),
    ("CMA-ES", "margins_cma_es"),
    ("Nelder-Mead", "margins_nelder_mead"),
    ("ensemble-refined", "margins_ensemble"),
]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)


def bootstrap_p5(margins, rng):
    idx = rng.integers(0, len(margins), size=(BOOT, len(margins)))
    return np.percentile(np.percentile(margins[idx], 5, axis=1), [2.5, 97.5])


def main():
    d = np.load(ROOT / "experiments" / "e5b_result.npz")
    rng = np.random.default_rng(SEED)

    table = {}
    print("| design | purity (95% Wilson) | 5th-percentile margin, m (95% bootstrap) |")
    print("|---|---|---|")
    for name, key in DESIGNS:
        m = np.asarray(d[key], dtype=float)
        k, n = int((m > 0).sum()), len(m)
        lo, hi = wilson(k, n)
        p5 = float(np.percentile(m, 5))
        blo, bhi = bootstrap_p5(m, rng)
        table[name] = {
            "correct": f"{k}/{n}",
            "wilson_pct": [round(lo), round(hi)],
            "p5_margin_m": round(p5, 2),
            "p5_ci_m": [round(float(blo), 2), round(float(bhi), 2)],
        }
        print(f"| {name} | {k}/{n} ({round(lo)}–{round(hi)}%) | "
              f"{p5:+.2f} ({blo:+.2f} to {bhi:+.2f}) |")

    out = ROOT / "experiments" / "design_table.json"
    out.write_text(json.dumps({"bootstrap": BOOT, "seed": SEED, "designs": table},
                              indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out.name}: {len(table)} designs, {BOOT} bootstrap resamples, seed {SEED}")


if __name__ == "__main__":
    main()
