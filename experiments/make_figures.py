"""Generate the three writeup figures into docs/figures/.

Style: light-surface paper figures; palette and mark specs follow the repo's
figure conventions (thin 2px lines, hairline grid, direct labels, one axis).
"""

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
FIGS.mkdir(parents=True, exist_ok=True)

# palette (light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TERRAIN = "#e1e0d9"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": SECONDARY,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
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
V0_OPT, E_OPT = np.array([2.40641, 0.47038]), 0.68051
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
    ax.annotate("initial guess", (1.35, 0.66), ha="center", color=ORANGE, fontsize=9,
                xytext=(0, 8), textcoords="offset points")
    ax.annotate("optimized", (2.85, 0.32), ha="center", color=BLUE, fontsize=9,
                xytext=(0, 8), textcoords="offset points")
    ax.set_xlim(-0.1, 5.2)
    ax.set_ylim(-0.08, 1.45)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("E1 — inverse design through two composed Tesseracts", loc="left")
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
    ax.set_title("E1 — distance to cup")
    ax.annotate(f"{hist[-1, 1]*100:.1f} cm", (len(hist) - 1, hist[-1, 1]),
                textcoords="offset points", xytext=(-8, 10), ha="right", color=BLUE, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGS / "e1_convergence.png")
    plt.close(fig)
    print("wrote e1_convergence.png")


def fig_e3():
    # values from experiments/e3_naive_vs_saltation.py (deterministic)
    dts = np.array([4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5])
    naive = np.zeros_like(dts)
    truth = 0.09037774
    fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=200)
    ax.axhline(truth, color=BLUE, lw=2.0)
    ax.annotate("true gradient (saltation, = FD)", (dts[2], truth),
                textcoords="offset points", xytext=(0, 7), ha="center", color=BLUE, fontsize=9)
    ax.plot(dts, naive, color=ORANGE, lw=2.0, marker="o", ms=5, markeredgecolor=SURFACE, markeredgewidth=0.8)
    ax.annotate("naive autodiff through time-stepper", (dts[2], 0.0),
                textcoords="offset points", xytext=(0, -16), ha="center", color=ORANGE, fontsize=9)
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_ylim(-0.03, 0.115)
    ax.set_xlabel("integrator step dt  (refining →)")
    ax.set_ylabel(r"d x(T) / d v$_{0y}$")
    ax.set_title("E3 — naive gradient never converges (O(1) event bias)", loc="left")
    fig.tight_layout()
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
    ax.annotate("initial terrain", (0.55, 0.19), color=MUTED, fontsize=8)
    for r, c, label, cup in ((r_slow, BLUE, "slow inlet (1.6 m/s)", 2.6), (r_fast, ORANGE, "fast inlet (2.6 m/s)", 4.6)):
        traj = np.asarray(r["traj"])
        ax.plot(traj[:, 1], traj[:, 2], color=c, lw=2.0, zorder=3)
        imp = np.asarray(r["impact_x"])[: int(r["n_events"])]
        ax.scatter(imp, h_des(imp), s=18, color=c, zorder=4, edgecolors=SURFACE, linewidths=0.8)
        cy = float(h_des(cup))
        ax.scatter([cup], [cy], marker="v", s=70, color=c, zorder=5)
        ax.annotate(f"cup {'A' if c == BLUE else 'B'}", (cup, cy), textcoords="offset points",
                    xytext=(0, -16), ha="center", color=c, fontsize=9)
    ax.annotate("slow inlet", (1.05, 0.72), color=BLUE, fontsize=9)
    ax.annotate("fast inlet", (2.30, 0.60), color=ORANGE, fontsize=9)
    ax.set_xlim(-0.1, 5.4)
    ax.set_ylim(-0.08, 1.45)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("E4 — passive sorter: one designed terrain routes two inlet speeds to two cups", loc="left")
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
    ax.set_title("E2b — NUTS posterior through the solver", loc="left")
    fig.tight_layout()
    fig.savefig(FIGS / "e2b_posterior.png")
    plt.close(fig)
    print("wrote e2b_posterior.png")


def main():
    t = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    fig_trajectory(t)
    fig_convergence()
    fig_e3()
    fig_e4(t)
    fig_e2b()


if __name__ == "__main__":
    main()
