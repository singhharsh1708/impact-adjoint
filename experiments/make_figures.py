"""Generate the three writeup figures into docs/figures/.

Style: light-surface paper figures; palette and mark specs follow the repo's
figure conventions (thin 2px lines, hairline grid, direct labels, one axis).
"""

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
FIGS.mkdir(parents=True, exist_ok=True)

# palette (light mode)
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
AQUA_TEXT = "#137d57"
TERRAIN = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY,
    "xtick.color": TICK_TEXT,
    "ytick.color": TICK_TEXT,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "font.size": 9,
})

TERRAIN_P = {"amp": np.array([0.2, 0.1, 0.15]), "ctr": np.array([1.0, 2.5, 4.0]), "wid": np.array([0.5, 0.4, 0.6])}
FIXED = {**TERRAIN_P, "y0": 1.0, "mu": 0.1, "drag": 0.0, "t_final": 2.0, "dt": 1e-3, "n_samples": 1200, "v_stop": 1e-4}
V0_INIT, E_INIT = np.array([2.0, 0.5]), 0.7
_E1P = np.load(ROOT / "experiments" / "e1_params_history.npy")[-1]
V0_OPT, E_OPT = _E1P[:2], float(_E1P[2])
TARGET_X = 4.3


def h_of(x):
    x = np.asarray(x)[..., None]
    return np.sum(TERRAIN_P["amp"] * np.exp(-((x - TERRAIN_P["ctr"]) ** 2) / (2 * TERRAIN_P["wid"] ** 2)), axis=-1)


def fig_trajectory(t):
    r0 = t.apply({**FIXED, "v0": V0_INIT, "e": E_INIT})
    r1 = t.apply({**FIXED, "v0": V0_OPT, "e": E_OPT})
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=200)
    xs = np.linspace(-0.1, 5.2, 600)
    ax.fill_between(xs, -0.08, h_of(xs), color=TERRAIN, lw=0, zorder=1)
    ax.plot(xs, h_of(xs), color=BASELINE, lw=1.2, zorder=2)
    for r, c, label in ((r0, ORANGE, "initial guess"), (r1, BLUE, "optimized")):
        traj = np.asarray(r["traj"])
        nev = int(r["n_events"])
        ax.plot(traj[:, 1], traj[:, 2], color=c, lw=2.0, zorder=3, label=label)
        imp = np.asarray(r["impact_x"])[:nev]
        ax.scatter(imp, h_of(imp), s=18, color=c, zorder=4, edgecolors=SURFACE, linewidths=0.8)
    cup_y = float(h_of(TARGET_X))
    ax.scatter([TARGET_X], [cup_y], marker="v", s=70, color=INK, zorder=5)
    ax.annotate("cup", (TARGET_X, cup_y), textcoords="offset points", xytext=(8, 6),
                ha="left", color=INK, fontsize=9)
    ax.annotate("initial guess", (1.35, 0.66), ha="center", color=ORANGE_TEXT, fontsize=9,
                xytext=(0, 8), textcoords="offset points")
    ax.annotate("optimized", (2.85, 0.32), ha="center", color=BLUE_TEXT, fontsize=9,
                xytext=(0, 8), textcoords="offset points")
    ax.set_xlim(-0.1, 5.2)
    ax.set_ylim(-0.08, 1.45)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("E1: inverse design through two composed Tesseracts", loc="left")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "e1_trajectory.png")
    plt.close(fig)
    print("wrote e1_trajectory.png  (initial bounces:", int(r0["n_events"]), "optimized:", int(r1["n_events"]), ")")


def fig_convergence():
    hist = np.load(ROOT / "experiments" / "e1_history.npy")  # (loss, miss)
    fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=200)
    ax.plot(np.arange(len(hist)), hist[:, 1], color=BLUE, lw=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("Adam iteration")
    ax.set_ylabel("miss distance [m]")
    ax.set_title("E1: distance to cup", loc="left")
    ax.annotate(f"{hist[-1, 1]*100:.1f} cm", (len(hist) - 1, hist[-1, 1]),
                textcoords="offset points", xytext=(-8, 10), ha="right", color=BLUE_TEXT, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "e1_convergence.png")
    plt.close(fig)
    print("wrote e1_convergence.png")


def fig_e3():
    # wider two-panel figure: scale fonts so rendered text matches the set
    with plt.rc_context({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11}):
        return _fig_e3_inner()


def _fig_e3_inner():
    rows = np.load(ROOT / "experiments" / "e3_rows.npy")
    dts, naive, interp = rows[:, 0], rows[:, 1], rows[:, 2]
    # measured, not retyped: the mean of the per-dt saltation column, which
    # spans 6e-14 across the whole sweep
    if rows.shape[1] <= 3:
        raise SystemExit("e3_rows.npy has no truth column; re-run e3_naive_vs_saltation.py")
    truth = float(rows[:, 3].mean())
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.0), dpi=200)

    ax.axhline(truth, color=BLUE, lw=2.0)
    ax.annotate("true gradient (saltation, = FD)", (dts[2], truth),
                textcoords="offset points", xytext=(0, 7), ha="center", color=BLUE_TEXT, fontsize=9)
    ax.plot(dts, naive, color=ORANGE, lw=2.0, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.annotate("grid-reset autodiff (exactly 0)", (dts[2], 0.0),
                textcoords="offset points", xytext=(0, -16), ha="center", color=ORANGE_TEXT, fontsize=9)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(-0.03, 0.115)
    ax.set_xlabel("integrator step dt  (refining →)")
    ax.set_ylabel(r"d x(T) / d v$_{0y}$")
    ax.set_title("what jax.grad returns", loc="left")

    rel_interp = np.abs(interp - truth) / truth
    ax2.plot(dts, rel_interp, color=AQUA, lw=1.8, marker="o", ms=4, markeredgecolor=SURFACE, markeredgewidth=0.7)
    ax2.annotate("hand-interpolated event (converging, erratic)", (dts[3], rel_interp[3]),
                 textcoords="offset points", xytext=(0, -22), ha="center",
                 color=AQUA_TEXT, fontsize=9)
    # measured per-dt saltation, not a literal: column 3 of e3_rows.npy
    salt_rel = np.abs(rows[:, 3] - rows[:, 3].mean()) / np.abs(rows[:, 3].mean())
    ax2.plot(dts, np.maximum(salt_rel, 1e-17), color=BLUE, lw=1.8, marker="s", ms=4,
             markeredgecolor=SURFACE, markeredgewidth=0.7)
    ax2.annotate("saltation (relative spread below 1e-12)", (dts[3], 1e-15),
                 textcoords="offset points", xytext=(0, 7), ha="center", color=BLUE_TEXT, fontsize=9)
    ax2.axhline(1.0, color=ORANGE, lw=2.0)
    ax2.annotate("grid-reset (100% bias)", (dts[3], 1.0),
                 textcoords="offset points", xytext=(0, 6), ha="center", color=ORANGE_TEXT, fontsize=9)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.invert_xaxis()
    ax2.set_ylim(1e-16, 3e1)
    ax2.set_xlabel("integrator step dt  (refining →)")
    ax2.set_ylabel("relative gradient error")
    ax2.set_title("error vs refinement", loc="left")

    fig.suptitle("E3: what autodiff sees at an impact", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIGS / "e3_bias.png")
    plt.close(fig)
    print("wrote e3_bias.png")


def fig_e4(t):
    d = np.load(ROOT / "experiments" / "e4_result.npz")
    des = {"amp": d["amp"], "ctr": d["ctr"], "wid": d["wid"]}

    def h_des(x):
        x = np.asarray(x)[..., None]
        return np.sum(des["amp"] * np.exp(-((x - des["ctr"]) ** 2) / (2 * des["wid"] ** 2)), axis=-1)

    fixed = {"y0": 1.0, "e": 0.7, "mu": 0.1, "drag": 0.0, "t_final": 2.2, "dt": 1e-3, "n_samples": 1200}
    r_slow = t.apply({**fixed, **des, "v0": np.array([1.6, 0.3])})
    r_fast = t.apply({**fixed, **des, "v0": np.array([2.6, 0.3])})

    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=200)
    xs = np.linspace(-0.1, 5.4, 600)
    ax.fill_between(xs, -0.08, h_des(xs), color=TERRAIN, lw=0, zorder=1)
    ax.plot(xs, h_des(xs), color=BASELINE, lw=1.2, zorder=2)
    h0 = 0.15 * np.sum(np.exp(-((xs[:, None] - np.array([1.2, 2.4, 3.6])) ** 2) / (2 * 0.5**2)), axis=1)
    ax.plot(xs, h0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("initial terrain", (0.55, 0.19), color=TICK_TEXT, fontsize=8)
    for r, c, label, cup in ((r_slow, BLUE, "slow inlet (1.6 m/s)", 2.6), (r_fast, ORANGE, "fast inlet (2.6 m/s)", 4.6)):
        traj = np.asarray(r["traj"])
        ax.plot(traj[:, 1], traj[:, 2], color=c, lw=2.0, zorder=3)
        imp = np.asarray(r["impact_x"])[: int(r["n_events"])]
        ax.scatter(imp, h_des(imp), s=18, color=c, zorder=4, edgecolors=SURFACE, linewidths=0.8)
        cy = float(h_des(cup))
        ax.scatter([cup], [cy], marker="v", s=70, color=c, zorder=5)
        ax.annotate(f"cup {'A' if c == BLUE else 'B'}", (cup, cy), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=9,
                    color=BLUE_TEXT if c == BLUE else ORANGE_TEXT)
    ax.annotate("slow inlet", (1.05, 0.72), color=BLUE_TEXT, fontsize=9)
    ax.annotate("fast inlet", (2.30, 0.60), color=ORANGE_TEXT, fontsize=9)
    ax.set_xlim(-0.1, 5.4)
    ax.set_ylim(-0.08, 1.45)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("E4: passive sorter: one designed terrain routes two inlet speeds to two cups", loc="left")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(FIGS / "e4_sorter.png")
    plt.close(fig)
    print("wrote e4_sorter.png  (slow bounces:", int(r_slow["n_events"]), "fast:", int(r_fast["n_events"]), ")")


def fig_e2b():
    d = np.load(ROOT / "experiments" / "e2b_posterior.npz")
    e_s, mu_s = d["e"], d["mu"]
    fig, ax = plt.subplots(figsize=(4.6, 3.4), dpi=200)
    ax.scatter(e_s, mu_s, s=9, color=BLUE, alpha=0.35, lw=0, zorder=3)
    ax.scatter([0.7], [0.1], marker="x", s=90, color=INK, lw=2.2, zorder=4)
    ax.annotate("truth", (0.7, 0.1), textcoords="offset points", xytext=(8, 6), color=INK, fontsize=10)
    ax.annotate(
        f"posterior\ne = {e_s.mean():.3f} ± {e_s.std():.3f}\nμ = {mu_s.mean():.3f} ± {mu_s.std():.3f}",
        (0.03, 0.96), xycoords="axes fraction", va="top", color=SECONDARY, fontsize=9,
    )
    ax.set_xlabel("restitution e")
    ax.set_ylabel("tangential loss μ")
    ax.set_title("E2b: NUTS posterior through the solver", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "e2b_posterior.png")
    plt.close(fig)
    print("wrote e2b_posterior.png")


def fig_e6():
    d = np.load(ROOT / "experiments" / "e6_result.npz")
    es, xs, x_mid = d["es"], d["xs"], float(d["x_mid"])
    fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=200)
    ax.axhspan(x_mid, 6.0, color="#f7ede8", lw=0, zorder=0)
    ax.axhspan(-0.5, x_mid, color="#e9f1fb", lw=0, zorder=0)
    ax.axhline(x_mid, color=BASELINE, lw=1.0, ls=(0, (4, 3)))
    in_dom = es <= 0.8751
    a = xs < x_mid
    ax.scatter(es[in_dom & a], xs[in_dom & a], s=26, color=BLUE, zorder=3)
    ax.scatter(es[in_dom & ~a], xs[in_dom & ~a], s=26, color=ORANGE, zorder=3)
    ax.scatter(es[~in_dom], xs[~in_dom], s=30, facecolors="none", edgecolors=MUTED, linewidths=1.4, zorder=3)
    for e_t in (0.5, 0.8):
        i = int(np.argmin(np.abs(es - e_t)))
        ax.scatter([es[i]], [xs[i]], s=120, facecolors="none", edgecolors=INK, linewidths=1.2, zorder=4)
        ax.annotate("trained", (es[i], xs[i]), textcoords="offset points", xytext=(0, 10),
                    ha="center", color=INK, fontsize=8)
    ax.annotate("bin A side", (0.365, 1.05), color=BLUE_TEXT, fontsize=9)
    ax.annotate("bin B side", (0.365, 5.15), color=ORANGE_TEXT, fontsize=9)
    ax.annotate("outside validated domain\n(superball rebound)", (0.9, 1.95), ha="center",
                color=TICK_TEXT, fontsize=8)
    ax.set_xlabel("restitution e  (trained on 0.5 and 0.8 only)")
    ax.set_ylabel("landing x [m]")
    ax.set_ylim(-0.3, 6.0)
    ax.set_title("E6: zero-shot: one threshold sorts the whole continuum", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "e6_generalization.png")
    plt.close(fig)
    print("wrote e6_generalization.png")


def main():
    t = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    fig_trajectory(t)
    fig_convergence()
    fig_e3()
    fig_e4(t)
    fig_e2b()
    fig_e6()


if __name__ == "__main__":
    main()
