"""Figures for the verification studies: convergence, gradient accuracy,
cost scaling, optimizer benchmark, robustness and generalization statistics.

Reads only committed npz artifacts, so it re-renders without rerunning any
study.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "docs" / "figures"
_rendered = {}
E = ROOT / "experiments"

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

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY, "xtick.color": TICK_TEXT, "ytick.color": TICK_TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
})


def fig_verification():
    c = np.load(E / "convergence_result.npz")
    g = np.load(E / "gradient_accuracy_result.npz")
    s = np.load(E / "scaling_result.npz")

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), dpi=200)

    ax = axes[0]
    ax.loglog(c["dts"], np.maximum(c["err_flat"], 1e-16), "o-", color=BLUE, ms=4, lw=1.8)
    ax.loglog(c["dts"], np.maximum(c["err_drag"], 1e-16), "s-", color=ORANGE, ms=4, lw=1.8)
    ref = c["dts"] ** 4 * (c["err_drag"][0] / c["dts"][0] ** 4)
    ax.loglog(c["dts"], ref, ls=(0, (4, 3)), color=MUTED, lw=1.2)
    ax.annotate("slope 4", (c["dts"][1], ref[1]), color=TICK_TEXT, fontsize=10,
                textcoords="offset points", xytext=(10, -6))
    ax.annotate("multi-bounce vs closed form", (c["dts"][5], c["err_flat"][5]), color=BLUE_TEXT,
                fontsize=9, textcoords="offset points", xytext=(0, 9), ha="center")
    ax.annotate("smooth arc vs analytic", (c["dts"][5], c["err_drag"][5]), color=ORANGE_TEXT,
                fontsize=9, textcoords="offset points", xytext=(0, -20), ha="center")
    ax.set_xlabel("integrator step dt")
    ax.set_ylabel("max abs error in qf")
    ax.set_title(f"convergence (order {float(c['order_drag']):.2f})", loc="left")

    ax = axes[1]
    for k, col in zip([k for k in g.files if k.startswith("rel_")],
                      (BLUE, ORANGE, AQUA, SECONDARY)):
        ax.loglog(g["hs"], np.maximum(g[k], 1e-17), "-", color=col, lw=1.6,
                  label=k.replace("rel_", ""))
    ax.set_xlabel("finite-difference step h")
    ax.set_ylabel("relative disagreement")
    ax.set_title("gradient vs finite differences", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)

    ax = axes[2]
    n = s["n_params"]
    ax.plot(n, s["t_vjp"], "o-", color=BLUE, ms=4, lw=1.8)
    ax.plot(n, s["t_apply"], "s-", color=ORANGE, ms=4, lw=1.8)
    fit = float(s["a_vjp"]) + float(s["b_vjp"]) * n
    ax.plot(n, fit, ls=(0, (4, 3)), color=MUTED, lw=1.2)
    ax.annotate(f"{float(s['b_vjp'])*1000:.0f} us per parameter", (n[-2], fit[-2]),
                color=TICK_TEXT, fontsize=9, textcoords="offset points", xytext=(-10, 12), ha="right")
    ax.annotate("VJP", (n[-1], s["t_vjp"][-1]), color=BLUE_TEXT, fontsize=10,
                textcoords="offset points", xytext=(-30, 6))
    ax.annotate("apply", (n[-1], s["t_apply"][-1]), color=ORANGE_TEXT, fontsize=10,
                textcoords="offset points", xytext=(-34, 6))
    ax.set_xlabel("number of parameters")
    ax.set_ylabel("time per call [ms]")
    ax.set_title(f"cost scaling (R2 = {float(s['r2_vjp']):.3f})", loc="left")

    fig.suptitle("Verification studies: the solver converges, the gradient is right, the cost is affine",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(FIGS / "study_verification.png")
    plt.close(fig)
    print("wrote study_verification.png")


def fig_benchmark():
    b = np.load(E / "optimizer_benchmark.npz")
    grid = b["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), dpi=200, sharey=True)

    def band(ax, key, color, label):
        c = np.maximum(b[key], 1e-9)
        med = np.nanmedian(c, axis=0)
        lo, hi = np.nanpercentile(c, 25, axis=0), np.nanpercentile(c, 75, axis=0)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(grid, med, color=color, lw=2.0, label=label)

    for ax, akey, title in ((axes[0], "adam_eval", "gradient charged as 2 solves"),
                            (axes[1], "adam_wall", "gradient charged at measured cost")):
        band(ax, akey, BLUE, "Adam on saltation gradients")
        band(ax, "cma", SECONDARY, "CMA-ES (tuned)")
        band(ax, "nm", MUTED, "Nelder-Mead")
        ax.set_yscale("log")
        ax.set_xlabel("budget [forward-solve units]")
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].set_ylabel("best objective")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")

    fig.suptitle(f"24-dim design, {len(b['seeds'])} random starts: median and interquartile band",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "study_optimizers.png")
    plt.close(fig)
    print("wrote study_optimizers.png")


def fig_stats():
    r = np.load(E / "robustness_stats.npz")
    g = np.load(E / "generalization_stats.npz")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), dpi=200)

    ax = axes[0]
    bins = np.linspace(min(r["point_margins"].min(), r["robust_margins"].min()),
                       max(r["point_margins"].max(), r["robust_margins"].max()), 45)
    ax.hist(r["point_margins"], bins=bins, color=ORANGE, alpha=0.55, label="point design")
    ax.hist(r["robust_margins"], bins=bins, color=BLUE, alpha=0.55, label="ensemble design")
    ax.axvline(0.0, color=INK, lw=1.2)
    ax.annotate("wrong bin", (0, ax.get_ylim()[1] * 0.62), color=INK, fontsize=9,
                textcoords="offset points", xytext=(-6, 0), ha="right")
    pk, pn = int(r["point_k"]), int(r["point_n"])
    rk, rn = int(r["robust_k"]), int(r["robust_n"])
    ax.set_xlabel("separation margin [m]  (negative = misclassified)")
    ax.set_ylabel("particles")
    ax.set_title(f"held-out purity {pk}/{pn} vs {rk}/{rn}", loc="left")
    # Record what the picture asserts, so a figure left behind by an artifact
    # change is caught. Byte-comparing the PNG cannot do this across platforms.
    _rendered["study_robustness.png"] = {
        "point_correct": f"{pk}/{pn}", "robust_correct": f"{rk}/{rn}"
    }
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    es = g["es"]
    ax.plot(es, g["point_frac_a"], "o-", color=ORANGE, ms=4, lw=1.8, label="point design")
    ax.plot(es, g["robust_frac_a"], "s-", color=BLUE, ms=4, lw=1.8, label="ensemble design")
    ax.fill_between(es, g["point_ci_lo"], g["point_ci_hi"], color=ORANGE, alpha=0.15, lw=0)
    ax.fill_between(es, g["robust_ci_lo"], g["robust_ci_hi"], color=BLUE, alpha=0.15, lw=0)
    for tr in (0.5, 0.8):
        ax.axvline(tr, color=MUTED, ls=(0, (2, 3)), lw=1.0)
    ax.annotate("trained", (0.5, 1.06), color=TICK_TEXT, fontsize=9, ha="center")
    ax.annotate("trained", (0.8, 1.06), color=TICK_TEXT, fontsize=9, ha="center")
    ax.set_ylim(-0.08, 1.15)
    ax.set_xlabel("restitution e")
    ax.set_ylabel("P(sorted to bin A)")
    ax.set_title(f"classification under {float(g['v_sd'])*100:.0f} cm/s inlet jitter", loc="left")
    ax.legend(frameon=False, fontsize=9, loc="center left")

    fig.suptitle("Robustness: the ensemble objective buys purity, the low tail and decisiveness under jitter, at some cost to the median",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(FIGS / "study_robustness.png")
    plt.close(fig)
    print("wrote study_robustness.png")



def _dump_rendered():
    import json
    (FIGS / "rendered_claims.json").write_text(
        json.dumps(_rendered, indent=2, sort_keys=True) + "\n"
    )

if __name__ == "__main__":
    fig_verification()
    fig_stats()
    if (E / "optimizer_benchmark.npz").exists():
        fig_benchmark()
    _dump_rendered()
