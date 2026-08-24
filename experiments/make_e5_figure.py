"""E5 figure: designed separator profile + head-to-head convergence."""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tesseract_core import Tesseract

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
# tick labels are text: MUTED is 3.59:1 on white, under AA
TICK_TEXT = "#6a6964"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
# darkened only where the colour carries text: the series colours are
# 4.42:1 and 3.20:1 on white, under the 4.5:1 WCAG AA text threshold
BLUE_TEXT = "#1f66bd"
ORANGE_TEXT = "#c4501f"
AQUA = "#1baf7a"
TERRAIN = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY, "xtick.color": TICK_TEXT, "ytick.color": TICK_TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 12,
    "axes.titlesize": 13, "axes.labelsize": 12,
})

d = np.load(ROOT / "experiments" / "e5_result.npz")
CTR, WID = d["ctr"], d["wid"]
AMP = d["adam_amp"]
# Import the design's configuration rather than restating it. The restated
# copy had silently dropped v_stop, so the published figure re-solved under the
# schema default instead of the value the design declares.
import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import (  # noqa: E402
    BIN_PET, BIN_RUBBER, E_PET, E_RUBBER, V0,
)
from e5_separator import FIXED as _FIXED  # noqa: E402

# the figure draws trajectories, so it needs sampled points; everything else is
# the design's own configuration
FIXED = {**_FIXED, "n_samples": 1400}


def h_of(x):
    return np.sum(AMP * np.exp(-((np.asarray(x)[..., None] - CTR) ** 2) / (2 * WID**2)), axis=-1)


def main():
    t = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    r_rub = t.apply({**FIXED, "v0": V0, "e": E_RUBBER, "amp": AMP, "ctr": CTR, "wid": WID})
    r_pet = t.apply({**FIXED, "v0": V0, "e": E_PET, "amp": AMP, "ctr": CTR, "wid": WID})

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.6, 3.3), dpi=200, width_ratios=[1.5, 1.0])

    xs = np.linspace(-0.1, 5.6, 700)
    ax.fill_between(xs, -0.08, h_of(xs), color=TERRAIN, lw=0, zorder=1)
    ax.plot(xs, h_of(xs), color=BASELINE, lw=1.2, zorder=2)
    for r, c, cup, mat in ((r_rub, BLUE, BIN_RUBBER, 'e = 0.5 "rubber"'), (r_pet, ORANGE, BIN_PET, 'e = 0.8 "PET"')):
        traj = np.asarray(r["traj"])
        ax.plot(traj[:, 1], traj[:, 2], color=c, lw=2.0, zorder=3)
        imp = np.asarray(r["impact_x"])[: int(r["n_events"])]
        ax.scatter(imp, h_of(imp), s=16, color=c, zorder=4, edgecolors=SURFACE, linewidths=0.8)
        cy = float(h_of(cup))
        ax.scatter([cup], [cy], marker="v", s=70, color=c, zorder=5)
        ax.annotate(f"bin {'A' if c == BLUE else 'B'}", (cup, cy), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=11,
                    color=BLUE_TEXT if c == BLUE else ORANGE_TEXT)
    ax.annotate('e = 0.5 "rubber"', (0.9, 0.75), color=BLUE_TEXT, fontsize=11)
    ax.annotate('e = 0.8 "PET"', (2.6, 0.62), color=ORANGE_TEXT, fontsize=11)
    ax.set_xlim(-0.1, 5.6)
    ax.set_ylim(-0.16, 1.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("designed terrain, two materials", loc="left", fontsize=12)
    ax.grid(axis="y")

    for key, c, ls in (("adam", BLUE, "-"),
                       ("cma-es", SECONDARY, (0, (5, 2))),
                       ("nelder-mead", MUTED, (0, (2, 2)))):
        tr = d[f"{key}_trace"]
        ax2.plot(tr[:, 0], np.maximum(tr[:, 1], 1e-8), color=c, lw=2.0, ls=ls)
    ax2.annotate("Adam (ours)", (620, 3e-6), color=BLUE_TEXT, fontsize=11, ha="left")
    ax2.annotate("CMA-ES", (640, 6e-3), color=SECONDARY, fontsize=11)
    ax2.annotate("Nelder-Mead", (430, 1.7e-1), color=TICK_TEXT, fontsize=11)
    ax2.set_yscale("log")
    ax2.set_xlabel("solver evaluations (gradient call = 2)")
    ax2.set_ylabel("best objective")
    ax2.set_title("24-dim head-to-head", loc="left", fontsize=12)

    fig.suptitle("E5: bounce separator: same inlet, sorted by restitution", x=0.01, ha="left", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "e5_separator.png")
    plt.close(fig)
    print("wrote e5_separator.png  (rubber bounces:", int(r_rub["n_events"]), "PET:", int(r_pet["n_events"]),
          " misses:", round(abs(np.asarray(r_rub['qf'])[0] - BIN_RUBBER), 4), round(abs(np.asarray(r_pet['qf'])[0] - BIN_PET), 4), ")")


if __name__ == "__main__":
    main()
