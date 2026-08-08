"""Animate the E1 optimization: trajectory per Adam iterate converging to the cup.

Writes docs/figures/e1_optimization.gif (and a final-frame PNG for quick review).
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add("ForwardDiff")')

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tesseract_core import Tesseract

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
TERRAIN = "#e1e0d9"

TERRAIN_P = {"amp": np.array([0.2, 0.1, 0.15]), "ctr": np.array([1.0, 2.5, 4.0]), "wid": np.array([0.5, 0.4, 0.6])}
FIXED = {**TERRAIN_P, "y0": 1.0, "mu": 0.1, "drag": 0.0, "t_final": 2.0, "dt": 1e-3, "n_samples": 700, "v_stop": 1e-4}
TARGET_X = 4.3


def h_of(x):
    x = np.asarray(x)[..., None]
    return np.sum(TERRAIN_P["amp"] * np.exp(-((x - TERRAIN_P["ctr"]) ** 2) / (2 * TERRAIN_P["wid"] ** 2)), axis=-1)


def main():
    params = np.load(ROOT / "experiments" / "e1_params_history.npy")  # (150, 3)
    hist = np.load(ROOT / "experiments" / "e1_history.npy")  # (150, 2) loss, miss
    t = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")

    stride = 3
    iters = list(range(0, len(params), stride))
    if iters[-1] != len(params) - 1:
        iters.append(len(params) - 1)

    print(f"simulating {len(iters)} frames ...")
    trajs = []
    for i in iters:
        v0 = params[i, :2]
        e = float(params[i, 2])
        r = t.apply({**FIXED, "v0": v0, "e": e})
        trajs.append(np.asarray(r["traj"])[:, 1:3])

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": BASELINE,
        "axes.labelcolor": SECONDARY, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    })
    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=110)
    xs = np.linspace(-0.1, 5.2, 500)
    ax.fill_between(xs, -0.08, h_of(xs), color=TERRAIN, lw=0, zorder=1)
    ax.plot(xs, h_of(xs), color=BASELINE, lw=1.2, zorder=2)
    cup_y = float(h_of(TARGET_X))
    ax.scatter([TARGET_X], [cup_y], marker="v", s=80, color=INK, zorder=6)
    ax.annotate("cup", (TARGET_X, cup_y), textcoords="offset points", xytext=(8, 6), ha="left", color=INK)
    ax.set_xlim(-0.1, 5.2)
    ax.set_ylim(-0.08, 1.45)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Inverse design by gradient descent through two Tesseracts", loc="left")

    fig.tight_layout()

    n_ghost = 6
    ghosts = [ax.plot([], [], color=BLUE, lw=1.2, alpha=0.0, zorder=3)[0] for _ in range(n_ghost)]
    (line,) = ax.plot([], [], color=BLUE, lw=2.2, zorder=5)
    info = ax.text(0.015, 0.94, "", transform=ax.transAxes, va="top", color=SECONDARY, fontsize=10)

    hold = 10

    def update(frame):
        k = min(frame, len(iters) - 1)
        for g_idx, g in enumerate(ghosts):
            src = k - (n_ghost - g_idx)
            if src >= 0:
                g.set_data(trajs[src][:, 0], trajs[src][:, 1])
                g.set_alpha(0.06 + 0.05 * g_idx)
            else:
                g.set_alpha(0.0)
        line.set_data(trajs[k][:, 0], trajs[k][:, 1])
        it = iters[k]
        info.set_text(f"Adam iteration {it:3d}    miss {hist[it, 1]*100:6.1f} cm")
        return [*ghosts, line, info]

    anim = FuncAnimation(fig, update, frames=len(iters) + hold, interval=110, blit=True)
    out = FIGS / "e1_optimization.gif"
    anim.save(out, writer=PillowWriter(fps=9))
    update(len(iters) - 1)
    fig.savefig(FIGS / "e1_optimization_final.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) + final-frame PNG")


if __name__ == "__main__":
    main()
