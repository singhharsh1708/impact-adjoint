"""Randomized property check over the solver's declared input space.

The golden tests pin specific configurations and the oracles check specific
trajectories. Neither samples the space, so a configuration that violates an
invariant only somewhere in the interior would not be caught.

This draws random inputs from the ranges the schema declares and asserts the
properties the solver claims, rather than any particular number:

  * status is one of the three documented codes, and t_end == t_final exactly
    when status == 0
  * n_events never exceeds the event budget, and impact_x is exactly zero
    beyond n_events with no NaN anywhere
  * energy never increases across the run when the model is lossy, and is
    conserved along the whole sampled trajectory in the lossless case, which
    is drawn deliberately rather than hoped for
  * every Jacobian entry is finite
  * the analytic Jacobian agrees with a central difference through the solver
    at fixed event topology

Run with a seed for a reproducible sweep:  python scripts/fuzz_solver.py 500 7
"""

import sys
from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

from tesseract_core import Tesseract  # noqa: E402
from tesseract_core.runtime.core import load_module_from_path  # noqa: E402

ROOT = Path(__file__).parent.parent
API = ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py"
G = 9.81
# The event budget lives on the schema. Restating it would make the
# over-budget check below fire spuriously if the cap rose, or stop catching
# anything if it fell.
MAX_EVENTS = load_module_from_path(API).MAX_EVENTS


def draw(rng):
    nb = int(rng.integers(1, 5))
    # one draw in eight is the lossless case, which uniform(0.15, 1.0) would
    # otherwise never produce, and which is the only case with an exact
    # conservation law to check against
    lossless = rng.random() < 0.125
    return {
        "lossless": lossless,
        "v0": np.array([rng.uniform(0.5, 4.0), rng.uniform(-1.0, 2.0)]),
        "y0": float(rng.uniform(0.4, 2.0)),
        "e": 1.0 if lossless else float(rng.uniform(0.02, 1.0)),
        "mu": 0.0 if lossless else float(rng.uniform(0.0, 0.95)),
        "amp": rng.uniform(0.0, 0.35, nb),
        "ctr": np.sort(rng.uniform(0.3, 5.0, nb)),
        "wid": rng.uniform(0.2, 0.8, nb),
        "drag": 0.0 if lossless else float(rng.choice([0.0, 0.0, rng.uniform(0.0, 0.4)])),
        "t_final": float(rng.uniform(1.0, 2.5)),
        "dt": float(rng.choice([1e-3, 5e-4, 2e-4])),
        "n_samples": 400 if lossless else 0,
        "v_stop": 1e-4,
    }


def energy(q, cfg):
    x, y, vx, vy = q
    return 0.5 * (vx * vx + vy * vy) + G * y


PROBES = ["v0x", "v0y", "y0", "e", "mu", "amp"]


def _perturb(cfg, probe, h, k=0):
    """Return cfg shifted by +h along one input, and the column it maps to."""
    c = {**cfg}
    if probe == "v0x":
        c["v0"] = cfg["v0"] + np.array([h, 0.0]); return c, ("v0", 0)
    if probe == "v0y":
        c["v0"] = cfg["v0"] + np.array([0.0, h]); return c, ("v0", 1)
    if probe == "y0":
        c["y0"] = cfg["y0"] + h; return c, ("y0", None)
    if probe == "e":
        c["e"] = cfg["e"] + h; return c, ("e", None)
    if probe == "mu":
        c["mu"] = cfg["mu"] + h; return c, ("mu", None)
    amp = np.array(cfg["amp"], dtype=float); amp[k] += h
    c["amp"] = amp; return c, ("amp", k)


def check(t, cfg_full, rng, report):
    lossless = cfg_full.get("lossless", False)
    cfg = {k: v for k, v in cfg_full.items() if k != "lossless"}
    out = t.apply(cfg)
    qf = np.asarray(out["qf"], dtype=float)
    nev = int(out["n_events"])
    status = int(out["status"])
    t_end = float(out["t_end"])
    imp = np.asarray(out["impact_x"], dtype=float)

    if status not in (0, 1, 2):
        report("status outside {0,1,2}", cfg, f"status={status}")
    if not np.all(np.isfinite(qf)):
        report("non-finite qf", cfg, f"qf={qf}")
    if nev > MAX_EVENTS:
        report("n_events over budget", cfg, f"n_events={nev}")
    if not np.all(np.isfinite(imp)):
        report("non-finite impact_x", cfg, f"impact_x={imp}")
    if nev < len(imp) and np.any(imp[nev:] != 0.0):
        report("impact_x padding not exactly zero", cfg, f"tail={imp[nev:]}")
    if status == 0 and abs(t_end - cfg["t_final"]) > 1e-9:
        report("status 0 but t_end != t_final", cfg, f"t_end={t_end}")
    if status != 0 and t_end > cfg["t_final"] + 1e-9:
        report("t_end past t_final", cfg, f"t_end={t_end}")

    # energy: launch state to final state
    q0 = np.array([0.0, cfg["y0"], cfg["v0"][0], cfg["v0"][1]])
    e0, e1 = energy(q0, cfg), energy(qf, cfg)
    if not lossless and e1 > e0 + 1e-6:
        report("energy increased", cfg, f"{e0:.6f} -> {e1:.6f}")
    if lossless:
        # e = 1, mu = 0, no drag: energy is conserved exactly, so check it
        # along the whole sampled path rather than only at the two endpoints
        traj = np.asarray(out["traj"], dtype=float)
        if traj.size:
            ke = 0.5 * (traj[:, 3] ** 2 + traj[:, 4] ** 2) + G * traj[:, 2]
            drift = float(np.max(np.abs(ke - ke[0])) / abs(ke[0]))
            if drift > 1e-10:
                report("lossless energy not conserved along trajectory", cfg,
                       f"relative drift {drift:.2e} over {len(ke)} samples")

    jac = t.jacobian(cfg, jac_inputs={"v0", "e"}, jac_outputs={"qf"})
    for k, v in jac["qf"].items():
        a = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(a)):
            report(f"non-finite d qf/d {k}", cfg, f"{a}")

    # central difference at fixed event topology, on one input per draw so the
    # explicit parameter columns get swept too and not just v0y
    h = 1e-6
    # a central difference has to stay inside the declared domain: e is (0, 1]
    # and mu is [0, 1), so a draw sitting on either boundary cannot be probed
    # there. Restrict the choice rather than emit an out-of-range call.
    usable = [q for q in PROBES
              if not (q == "e" and cfg["e"] + h > 1.0)
              and not (q == "mu" and cfg["mu"] - h < 0.0)]
    probe = usable[int(rng.integers(len(usable)))]
    k = int(rng.integers(len(cfg["amp"])))
    cfg_p, (name, col) = _perturb(cfg, probe, h, k)
    cfg_m, _ = _perturb(cfg, probe, -h, k)
    jac1 = t.jacobian(cfg, jac_inputs={name}, jac_outputs={"qf"})
    op, om = t.apply(cfg_p), t.apply(cfg_m)
    if int(op["n_events"]) == int(om["n_events"]) == nev and status == 0:
        fd = (np.asarray(op["qf"], float) - np.asarray(om["qf"], float)) / (2 * h)
        a = np.asarray(jac1["qf"][name], float)
        an = a if col is None else a[:, col]
        an = np.asarray(an, float).reshape(-1)
        scale = max(1.0, np.max(np.abs(fd)))
        err = float(np.max(np.abs(an - fd)) / scale)
        if err > 2e-5:
            report(f"analytic vs FD disagreement on {probe}", cfg, f"rel={err:.2e}")
        return err, status
    return None, status


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = np.random.default_rng(seed)
    failures = []

    def report(what, cfg, detail):
        failures.append((what, detail, {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                        for k, v in cfg.items()}))

    t = Tesseract.from_tesseract_api(API)
    errs, checked, lossless_seen = [], 0, 0
    status_seen = {0: 0, 1: 0, 2: 0}
    for i in range(n):
        cfg = draw(rng)
        lossless_seen += int(cfg["lossless"])
        try:
            e, status = check(t, cfg, rng, report)
        except Exception as exc:  # a raised error is itself a finding
            report(f"raised {type(exc).__name__}", cfg, str(exc)[:160])
            continue
        checked += 1
        if e is not None:
            errs.append(e)
        status_seen[status] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n} drawn, {len(failures)} findings", flush=True)

    print(f"\nchecked {checked}/{n} configurations (seed {seed})")
    print(f"status coverage: 0={status_seen[0]} 1={status_seen[1]} 2={status_seen[2]}")
    print(f"FD-checked {len(errs)}/{checked - lossless_seen} eligible draws "
          f"({lossless_seen} lossless draws are budget-truncated by construction)")
    if errs:
        print(f"analytic-vs-FD worst {max(errs):.2e}, median {np.median(errs):.2e}, "
              f"over {len(errs)} topology-stable draws")
    if failures:
        print(f"\n{len(failures)} FINDINGS")
        for what, detail, cfg in failures[:5]:
            print(f"  - {what}: {detail}")
            print(f"    cfg={cfg}")
        raise SystemExit(1)
    # coverage is only meaningful once we know there were no violations to
    # report; checking it first would swallow them
    # lossless draws almost always exhaust the event budget, so they can never
    # reach the FD check; counting them in the denominator would make this
    # gate flake rather than measure coverage
    eligible = checked - lossless_seen
    if eligible and len(errs) < 0.15 * eligible:
        print(f"\nINSUFFICIENT COVERAGE: only {len(errs)}/{eligible} eligible draws "
              "were topology-stable enough to FD-check the adjoint; the sweep "
              "cannot vouch for the gradient")
        raise SystemExit(1)
    print("no invariant violations")


if __name__ == "__main__":
    main()
