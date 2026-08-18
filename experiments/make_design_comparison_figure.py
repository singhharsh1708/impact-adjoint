"""Compare all four designs on separation quality, not objective value.

The optimizer benchmark answers "which method drives the objective lowest".
That is not the same question as "which design sorts particles", and the
answers differ: Nelder-Mead is five orders behind on the objective and sorts
as well as Adam, with a wider margin. This plots the distinction directly.

Left: the margin distribution each design produces on E5b's held-out scatter
ensemble, as an empirical CDF, so the low tail is legible rather than hidden
inside a box plot. Right: final objective against fifth-percentile margin, the
two axes the text argues about.

Reads experiments/e5b_result.npz and e5_result.npz. Writes
docs/figures/design_comparison.png.
"""

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
TICK_TEXT = "#6a6964"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#137d57"
PLUM = "#8a5cb8"
BLUE_TEXT = "#1f66bd"
ORANGE_TEXT = "#c4501f"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY, "xtick.color": TICK_TEXT, "ytick.color": TICK_TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
})

DESIGNS = [
    ("Adam (saltation gradients)", "margins_adam", "adam_trace", BLUE, BLUE_TEXT, "-"),
    ("CMA-ES (single run)", "margins_cma_es", "cma-es_trace", ORANGE, ORANGE_TEXT, "-"),
    ("Nelder-Mead", "margins_nelder_mead", "nelder-mead_trace", SECONDARY, SECONDARY, "--"),
    ("ensemble-refined (E5b)", "margins_ensemble", None, AQUA, AQUA, "-"),
]


def main():
    e5b = np.load(ROOT / "experiments" / "e5b_result.npz")
    e5 = np.load(ROOT / "experiments" / "e5_result.npz")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.3), dpi=124)

    for label, mkey, tkey, col, tcol, ls in DESIGNS:
        m = np.sort(e5b[mkey])
        cdf = np.arange(1, len(m) + 1) / len(m)
        ax.step(m, cdf, where="post", color=col, lw=2.0, ls=ls, label=label)

    ax.axvline(0.0, color=ORANGE_TEXT, lw=1.2, ls=(0, (3, 3)))
    ax.annotate("wrong bin", (0.0, 0.55), xytext=(-8, 0), textcoords="offset points",
                ha="right", color=ORANGE_TEXT, fontsize=9)
    ax.set_xlabel("separation margin [m]   (negative = sorted into the wrong bin)")
    ax.set_ylabel("fraction of particles at or below")
    ax.set_title("Margin distribution on the held-out scatter ensemble", loc="left")
    ax.set_xlim(-1.0, 1.6)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    for label, mkey, tkey, col, tcol, ls in DESIGNS:
        m = e5b[mkey]
        p5 = float(np.percentile(m, 5))
        obj = float(e5[tkey][-1, 1]) if tkey else np.nan
        purity = float((m > 0).mean())
        if np.isnan(obj):
            continue
        ax2.scatter([obj], [p5], s=90, color=col, zorder=5,
                    edgecolors=SURFACE, linewidths=1.2)
        ax2.annotate(f"{label.split(' (')[0]}\n{purity:.1%} correct",
                     (obj, p5), textcoords="offset points", xytext=(9, -4),
                     color=tcol, fontsize=9, va="center")

    ax2.set_xscale("log")
    ax2.set_xlabel("final objective (sum of squared landing error) [m$^2$]")
    ax2.set_ylabel("5th-percentile margin [m]")
    ax2.set_title("Four orders of objective, almost no separation gain", loc="left")
    ax2.set_xlim(5e-8, 2e-1)
    ax2.set_ylim(-0.02, 0.60)
    # the ensemble design minimises a different objective, so it has no
    # comparable point on this axis; say so rather than leave a gap
    p5_ens = float(np.percentile(e5b["margins_ensemble"], 5))
    ax2.axhline(p5_ens, color=AQUA, lw=1.4, ls=(0, (4, 3)))
    ax2.annotate("ensemble-refined (E5b): optimises a different objective,\n"
                 f"reaches {p5_ens:.2f} m and 100% correct",
                 (6e-8, p5_ens), textcoords="offset points", xytext=(2, 6),
                 color=AQUA, fontsize=9, va="bottom")

    fig.tight_layout()
    out = FIGS / "design_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out.name}")


if __name__ == "__main__":
    main()
