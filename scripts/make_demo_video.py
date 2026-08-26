"""Build the demo video from the committed artifacts.

Nothing here is typed: every number on a card is read from results.json, and
the two motion clips are the animations the experiment scripts already
produce. Re-running this after the artifacts change produces a video that
agrees with them, the same property the figures have.

    python scripts/make_demo_video.py

Writes docs/demo.mp4. Needs ffmpeg on PATH.
"""

import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

ROOT = Path(__file__).parent.parent
FIGS = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "demo.mp4"

W, H, DPI, FPS = 1280, 720, 100, 24

INK = "#16161a"
SOFT = "#52514e"
PAPER = "#fcfcfb"
BLUE = "#2166c2"
TEAL = "#0f7a54"
ORANGE = "#b23c11"

R = json.loads((ROOT / "experiments" / "results.json").read_text())


def _fig():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
    fig.patch.set_facecolor(PAPER)
    return fig


def _clear(fig):
    fig.clf()
    fig.patch.set_facecolor(PAPER)
    return fig


def card(fig, writer, lines, seconds, align=0.5):
    """lines: list of (text, size, colour, weight)."""
    _clear(fig)
    n = len(lines)
    ys = np.linspace(0.72, 0.28, n) if n > 1 else [0.5]
    for (text, size, colour, weight), y in zip(lines, ys):
        fig.text(align, y, text, ha="center", va="center", fontsize=size,
                 color=colour, weight=weight, family="DejaVu Sans", wrap=True)
    for _ in range(int(seconds * FPS)):
        writer.grab_frame()


def image_card(fig, writer, png, caption, seconds, sub=None):
    _clear(fig)
    ax = fig.add_axes([0.06, 0.16, 0.88, 0.68])
    ax.imshow(mpimg.imread(FIGS / png))
    ax.axis("off")
    fig.text(0.5, 0.92, caption, ha="center", va="center", fontsize=21,
             color=INK, weight="bold")
    if sub:
        fig.text(0.5, 0.075, sub, ha="center", va="center", fontsize=14,
                 color=SOFT)
    for _ in range(int(seconds * FPS)):
        writer.grab_frame()


def clip(fig, writer, path, caption, sub=None, max_seconds=8):
    """Re-emit an existing animation's frames onto a captioned card."""
    tmp = ROOT / "_demo_frames"
    tmp.mkdir(exist_ok=True)
    for old in tmp.glob("*.png"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(path),
         "-vf", f"fps={FPS},scale=1000:-1", str(tmp / "f%04d.png")],
        check=True,
    )
    frames = sorted(tmp.glob("*.png"))[: int(max_seconds * FPS)]
    for f in frames:
        _clear(fig)
        ax = fig.add_axes([0.1, 0.16, 0.8, 0.66])
        ax.imshow(mpimg.imread(f))
        ax.axis("off")
        fig.text(0.5, 0.92, caption, ha="center", va="center", fontsize=21,
                 color=INK, weight="bold")
        if sub:
            fig.text(0.5, 0.075, sub, ha="center", va="center", fontsize=14,
                     color=SOFT)
        writer.grab_frame()
    for f in tmp.glob("*.png"):
        f.unlink()
    tmp.rmdir()


def terminal_card(fig, writer, title, lines, seconds):
    """Type out real captured output, line by line."""
    per_line = max(1, int(seconds * FPS / max(len(lines), 1)))
    for shown in range(1, len(lines) + 1):
        _clear(fig)
        fig.text(0.5, 0.92, title, ha="center", va="center", fontsize=21,
                 color=INK, weight="bold")
        ax = fig.add_axes([0.08, 0.12, 0.84, 0.72])
        ax.set_facecolor("#f4f4f1")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#e3e2dc")
        for i, ln in enumerate(lines[:shown]):
            colour = TEAL if ln.startswith("$") else INK
            ax.text(0.03, 0.93 - i * 0.105, ln, transform=ax.transAxes,
                    fontsize=13, family="DejaVu Sans Mono", color=colour,
                    va="top")
        for _ in range(per_line):
            writer.grab_frame()


def main():
    fig = _fig()
    writer = FFMpegWriter(fps=FPS, bitrate=2600,
                          metadata={"title": "impact-adjoint"})

    truth = R["E3_truth"]
    naive = R["E3_grid_reset_gradient"]
    adam, cma, nm = R["E5_adam"], R["E5_cma"], R["E5_nelder_mead"]
    orders = np.log10(nm / adam)

    with writer.saving(fig, str(OUT), dpi=DPI):
        card(fig, writer, [
            ("impact-adjoint", 52, INK, "bold"),
            ("End-to-end gradients through impact events", 26, SOFT, "normal"),
            ("Tesseract Hackathon 2026  ·  Track 1: inverse design", 18, BLUE, "normal"),
        ], 4.0)

        card(fig, writer, [
            ("Standard autodiff fails at contact.", 34, INK, "bold"),
            ("Not loudly. Silently.", 26, ORANGE, "normal"),
        ], 3.2)

        image_card(fig, writer, "e3_bias.png",
                   "The failure, measured",
                   6.5,
                   sub=(f"Grid-reset autodiff returns exactly {naive} at every step size.  "
                        f"The true derivative is +{truth}."))

        card(fig, writer, [
            ("The fix: saltation matrices", 34, INK, "bold"),
            ("Propagate sensitivity across the impact,", 22, SOFT, "normal"),
            ("in a Julia solver, under JAX's reverse-mode gradient.", 22, SOFT, "normal"),
        ], 4.0)

        card(fig, writer, [
            ("A real boundary", 34, INK, "bold"),
            ("Julia  ·  ForwardDiff dual numbers", 24, ORANGE, "normal"),
            ("↕   composed under one jax.grad", 20, SOFT, "normal"),
            ("JAX  ·  reverse-mode, Float32 and Float64", 24, BLUE, "normal"),
        ], 4.5)

        term = subprocess.run(
            ["python", str(ROOT / "scripts" / "proof_local.py")],
            capture_output=True, text=True, cwd=ROOT,
        )
        def _clip(line, n=68):
            line = line.rstrip()
            return line if len(line) <= n else line[: n - 1] + "\u2026"

        out_lines = [_clip(l) for l in term.stdout.strip().splitlines()
                     if l.strip()][-6:]
        terminal_card(fig, writer, "The boundary proof, run live",
                      ["$ python scripts/proof_local.py"] + out_lines, 6.5)

        clip(fig, writer, FIGS / "e1_optimization.gif",
             "E1  ·  inverse design through 5 bounces",
             sub=(f"Miss {R['E1_miss_start_m']:.2f} m → "
                  f"{R['E1_miss_final_m'] * 100:.1f} cm, across bounce-count changes."),
             max_seconds=6.5)

        clip(fig, writer, ROOT / "docs" / "site" / "_static" / "e5_learning.mp4",
             "E5  ·  a 24-dimensional resilience separator",
             sub="Two materials, one surface, sorted by restitution alone.",
             max_seconds=6.0)

        image_card(fig, writer, "study_optimizers.png",
                   "Gradients against gradient-free",
                   6.5,
                   sub=(f"At equal budget Adam reaches {adam:.1e} against "
                        f"CMA-ES {cma:.1e} and Nelder-Mead {nm:.1e}: "
                        f"{orders:.1f} orders."))

        image_card(fig, writer, "study_robustness.png",
                   "E5b  ·  design under uncertainty",
                   6.5,
                   sub=(f"Held-out purity {R['ROBUST_point_correct']} → "
                        f"{R['ROBUST_robust_correct']} over five independent ensembles "
                        f"(McNemar p = {R['ROBUST_mcnemar_p']:.4f})."))

        card(fig, writer, [
            ("Correctness: independent oracles, not self-agreement", 25, INK, "bold"),
            (f"Symbolic multi-bounce closed form:  {R['CLOSED_FORM_jacobian_worst']:.0e}", 21, SOFT, "normal"),
            (f"scipy reimplementation of the Jacobian:  {R['REFERENCE_jacobian_worst']:.0e}", 21, SOFT, "normal"),
            (f"Tesseract's own checker:  {R['CHECKGRAD_failures']} failures / {R['CHECKGRAD_checks']} checks", 21, SOFT, "normal"),
            (f"Observed order of accuracy:  {R['CONV_order']:.2f}", 21, SOFT, "normal"),
        ], 7.0)

        card(fig, writer, [
            ("Using Tesseract hard enough to break it", 28, INK, "bold"),
            ("Four fixes merged upstream into Pasteur Labs' repositories,", 21, SOFT, "normal"),
            ("two of them silent wrong-gradient bugs in Tesseract's own AD path.", 21, SOFT, "normal"),
            ("tesseract-core #667  ·  tesseract-jax #236  ·  mosaic #126, #141", 19, BLUE, "normal"),
        ], 6.0)

        card(fig, writer, [
            ("Every number regenerates from the committed artifacts.", 24, INK, "bold"),
            (f"The four verification checks re-run in {R['TIMING_checks_total_s']:.0f} seconds, warm.", 21, SOFT, "normal"),
            ("github.com/singhharsh1708/impact-adjoint", 22, BLUE, "normal"),
            ("impact-adjoint.vercel.app", 20, BLUE, "normal"),
        ], 6.0)

    plt.close(fig)
    size = OUT.stat().st_size / 1e6
    dur = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(OUT)],
        capture_output=True, text=True).stdout.strip()
    print(f"wrote {OUT.relative_to(ROOT)}: {float(dur):.1f} s, {size:.1f} MB")


if __name__ == "__main__":
    main()
