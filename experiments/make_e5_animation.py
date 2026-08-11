"""E5 animation: the surface learns to sort.

Re-runs the E5 Adam loop capturing the design at every iteration, then renders
the terrain morphing while the two materials' trajectories bifurcate into
their bins. Writes docs/figures/e5_learning.gif.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

import jax
import jax.numpy as jnp
import matplotlib
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tesseract_core import Tesseract

jax.config.update("jax_enable_x64", True)

import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import AMP0, AMP_MAX, BIN_PET, BIN_RUBBER, CTR, E_PET, E_RUBBER, FIXED, V0, WID, make_objective

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "docs" / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
TERRAIN = "#e1e0d9"


def h_of(x, amp):
    return np.sum(amp * np.exp(-((np.asarray(x)[..., None] - CTR) ** 2) / (2 * WID**2)), axis=-1)


def main():
    sim = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    score = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "score_target" / "tesseract_api.py")

    _, loss_and_grad, _ = make_objective(sim, score)
    amp = jnp.asarray(AMP0)
    opt = optax.adam(learning_rate=0.02)
    state = opt.init(amp)
    amps, losses = [np.asarray(amp)], []
    best = np.inf
    for _ in range(150):
        val, g = loss_and_grad(np.asarray(amp))
        best = min(best, val)   # report best-so-far, matching the headline number
        losses.append(best)
        updates, state = opt.update(jnp.asarray(g), state)
        amp = jnp.clip(optax.apply_updates(amp, updates), 0.0, AMP_MAX)
        amps.append(np.asarray(amp))
    losses.append(losses[-1])

    stride = 3
    idxs = list(range(0, len(amps), stride))
    if idxs[-1] != len(amps) - 1:
        idxs.append(len(amps) - 1)

    print(f"simulating {len(idxs)} frames x 2 materials ...")
    frames = []
    for i in idxs:
        a = amps[i]
        trajs = []
        for e in (E_RUBBER, E_PET):
            r = sim.apply({**FIXED, "n_samples": 700, "v0": V0, "e": e, "amp": a, "ctr": CTR, "wid": WID})
            trajs.append(np.asarray(r["traj"])[:, 1:3])
        frames.append((a, trajs))

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": BASELINE,
        "axes.labelcolor": SECONDARY, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    })
    fig, ax = plt.subplots(figsize=(7.2, 3.3), dpi=124)
    xs = np.linspace(-0.1, 5.6, 500)
    fill = [ax.fill_between(xs, -0.16, h_of(xs, frames[0][0]), color=TERRAIN, lw=0, zorder=1)]
    (tline,) = ax.plot(xs, h_of(xs, frames[0][0]), color=BASELINE, lw=1.2, zorder=2)
    (l_rub,) = ax.plot([], [], color=BLUE, lw=2.0, zorder=4)
    (l_pet,) = ax.plot([], [], color=ORANGE, lw=2.0, zorder=3)
    for cup, c, name in ((BIN_RUBBER, BLUE, "bin A"), (BIN_PET, ORANGE, "bin B")):
        ax.scatter([cup], [-0.06], marker="v", s=70, color=c, zorder=6)
        ax.annotate(name, (cup, -0.06), textcoords="offset points", xytext=(0, -14), ha="center", color=c)
    ax.annotate('e = 0.5 "rubber"', (0.35, 1.12), color=BLUE, fontsize=9)
    ax.annotate('e = 0.8 "PET"', (0.35, 1.00), color=ORANGE, fontsize=9)
    info = ax.text(0.99, 0.95, "", transform=ax.transAxes, va="top", ha="right", color=SECONDARY, fontsize=10)
    ax.set_xlim(-0.1, 5.6)
    ax.set_ylim(-0.16, 1.3)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("E5 — the surface learns to sort: one terrain, two materials, same inlet", loc="left")
    fig.tight_layout()

    hold = 12

    def update(frame):
        k = min(frame, len(idxs) - 1)
        a, trajs = frames[k]
        fill[0].remove()
        fill[0] = ax.fill_between(xs, -0.16, h_of(xs, a), color=TERRAIN, lw=0, zorder=1)
        tline.set_ydata(h_of(xs, a))
        l_rub.set_data(trajs[0][:, 0], trajs[0][:, 1])
        l_pet.set_data(trajs[1][:, 0], trajs[1][:, 1])
        it = idxs[k]
        info.set_text(f"Adam iteration {it:3d}    best objective {losses[it]:.2e}")
        return [fill[0], tline, l_rub, l_pet, info]

    anim = FuncAnimation(fig, update, frames=len(idxs) + hold, interval=120, blit=False)
    out = FIGS / "e5_learning.gif"
    anim.save(out, writer=PillowWriter(fps=8))
    plt.close(fig)
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
