"""E5b figure: held-out landing distributions, point vs robust design."""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tesseract_core import Tesseract

import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import BIN_PET, BIN_RUBBER, CTR, FIXED, WID
from e5b_robust_separator import N_TEST, X_MID, draw_ensemble

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 10,
})


def landings(t, amp, ensemble):
    xs, mats = [], []
    for v0, e, bx in ensemble:
        r = t.apply({**FIXED, "v0": np.asarray(v0), "e": e, "amp": np.asarray(amp), "ctr": CTR, "wid": WID})
        xs.append(float(np.asarray(r["qf"])[0]))
        mats.append(bx == BIN_RUBBER)
    return np.array(xs), np.array(mats)


def main():
    d = np.load(ROOT / "experiments" / "e5b_result.npz")
    t = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    test = draw_ensemble(np.random.default_rng(17), N_TEST)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.4), dpi=200, sharex=True)
    rng = np.random.default_rng(0)
    for ax, key, label, purity in ((axes[0], "point_amp", "point design (E5)", float(d["p_point_test"])),
                                   (axes[1], "robust_amp", "robust design (ensemble objective)", float(d["p_robust_test"]))):
        xs, is_rub = landings(t, d[key], test)
        jitter = rng.normal(0, 0.06, len(xs))
        ax.scatter(xs[is_rub], 0.5 + jitter[is_rub], s=9, color=BLUE, alpha=0.55, lw=0)
        ax.scatter(xs[~is_rub], -0.5 + jitter[~is_rub], s=9, color=ORANGE, alpha=0.55, lw=0)
        ax.axvline(X_MID, color=BASELINE, lw=1.0, ls=(0, (4, 3)))
        ax.set_yticks([0.5, -0.5], ["rubber", "PET"])
        ax.set_ylim(-1.1, 1.1)
        ax.set_title(f"{label}: held-out purity {purity:.1%}", loc="left", fontsize=10)
        n_wrong = int(np.sum((xs < X_MID) != is_rub))
        if n_wrong:
            ax.annotate(f"{n_wrong} misclassified", (X_MID, 0.0), textcoords="offset points",
                        xytext=(8, 0), color=SECONDARY, fontsize=8)
    axes[1].set_xlabel("landing x [m]  (200 held-out particles, v sd 5 cm/s, e sd 0.03)")
    fig.suptitle("E5b: design under uncertainty: sorting purity on held-out scatter", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGS / "e5b_purity.png")
    plt.close(fig)
    print("wrote e5b_purity.png")


if __name__ == "__main__":
    main()
