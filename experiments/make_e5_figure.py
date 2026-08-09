"""E5 figure: designed separator profile + head-to-head convergence."""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add("ForwardDiff")')

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
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
TERRAIN = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 10,
})

d = np.load(ROOT / "experiments" / "e5_result.npz")
CTR, WID = d["ctr"], d["wid"]
AMP = d["adam_amp"]
V0 = np.array([2.0, 0.4])
E_RUBBER, E_PET = 0.5, 0.8
BIN_RUBBER, BIN_PET = 2.8, 4.4
FIXED = {"y0": 1.0, "mu": 0.1, "drag": 0.0, "t_final": 2.2, "dt": 5e-4, "n_samples": 1400}


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
                    xytext=(0, -16), ha="center", color=c, fontsize=9)
    ax.annotate('e = 0.5 "rubber"', (0.9, 0.75), color=BLUE, fontsize=9)
    ax.annotate('e = 0.8 "PET"', (2.6, 0.62), color=ORANGE, fontsize=9)
    ax.set_xlim(-0.1, 5.6)
    ax.set_ylim(-0.16, 1.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("E5 — bounce separator: same inlet, sorted by restitution", loc="left")
    ax.grid(axis="y")

    for key, c, label in (("adam", BLUE, "Adam on saltation gradients"),
                          ("cma-es", ORANGE, "CMA-ES"),
                          ("nelder-mead", AQUA, "Nelder-Mead")):
        tr = d[f"{key}_trace"]
        ax2.plot(tr[:, 0], np.maximum(tr[:, 1], 1e-8), color=c, lw=2.0)
    ax2.annotate("Adam (ours)", (620, 3e-6), color=BLUE, fontsize=9, ha="left")
    ax2.annotate("CMA-ES", (640, 6e-3), color=ORANGE, fontsize=9)
    ax2.annotate("Nelder-Mead", (500, 6e-2), color=AQUA, fontsize=9)
    ax2.set_yscale("log")
    ax2.set_xlabel("solver evaluations (gradient call = 2)")
    ax2.set_ylabel("best objective")
    ax2.set_title("24-dim design: gradients vs gradient-free", loc="left")

    fig.tight_layout()
    fig.savefig(FIGS / "e5_separator.png")
    plt.close(fig)
    print("wrote e5_separator.png  (rubber bounces:", int(r_rub["n_events"]), "PET:", int(r_pet["n_events"]),
          " misses:", round(abs(np.asarray(r_rub['qf'])[0] - BIN_RUBBER), 4), round(abs(np.asarray(r_pet['qf'])[0] - BIN_PET), 4), ")")


if __name__ == "__main__":
    main()
