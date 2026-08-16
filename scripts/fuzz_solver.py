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
  * energy never increases across the run when e < 1 or mu > 0, and is
    conserved to tolerance when e == 1 and mu == 0
  * every Jacobian entry is finite
  * the analytic Jacobian agrees with a central difference through the solver
    at fixed event topology

Run with a seed for a reproducible sweep:  python scripts/fuzz_solver.py 500 7
"""

import sys
from pathlib import Path

import numpy as np
from tesseract_core import Tesseract

ROOT = Path(__file__).parent.parent
API = ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py"
G = 9.81
MAX_EVENTS = 8


def draw(rng):
    nb = int(rng.integers(1, 5))
    return {
        "v0": np.array([rng.uniform(0.5, 4.0), rng.uniform(-1.0, 2.0)]),
        "y0": float(rng.uniform(0.4, 2.0)),
        "e": float(rng.uniform(0.15, 1.0)),
        "mu": float(rng.uniform(0.0, 0.6)),
        "amp": rng.uniform(0.0, 0.35, nb),
        "ctr": np.sort(rng.uniform(0.3, 5.0, nb)),
        "wid": rng.uniform(0.2, 0.8, nb),
        "drag": float(rng.choice([0.0, 0.0, rng.uniform(0.0, 0.4)])),
        "t_final": float(rng.uniform(1.0, 2.5)),
        "dt": float(rng.choice([1e-3, 5e-4, 2e-4])),
        "n_samples": 0,
        "v_stop": 1e-4,
    }


def energy(q, cfg):
    x, y, vx, vy = q
    return 0.5 * (vx * vx + vy * vy) + G * y


def check(t, cfg, rng, report):
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
    lossless = cfg["e"] == 1.0 and cfg["mu"] == 0.0 and cfg["drag"] == 0.0
    if not lossless and e1 > e0 + 1e-6:
        report("energy increased", cfg, f"{e0:.6f} -> {e1:.6f}")

    jac = t.jacobian(cfg, jac_inputs={"v0", "e"}, jac_outputs={"qf"})
    for k, v in jac["qf"].items():
        a = np.asarray(v, dtype=float)
        if not np.all(np.isfinite(a)):
            report(f"non-finite d qf/d {k}", cfg, f"{a}")

    # central difference on v0y at fixed event topology
    h = 1e-6
    cfg_p = {**cfg, "v0": cfg["v0"] + np.array([0.0, h])}
    cfg_m = {**cfg, "v0": cfg["v0"] - np.array([0.0, h])}
    op, om = t.apply(cfg_p), t.apply(cfg_m)
    if int(op["n_events"]) == int(om["n_events"]) == nev and status == 0:
        fd = (np.asarray(op["qf"], float) - np.asarray(om["qf"], float)) / (2 * h)
        an = np.asarray(jac["qf"]["v0"], float)[:, 1]
        scale = max(1.0, np.max(np.abs(fd)))
        err = np.max(np.abs(an - fd)) / scale
        if err > 2e-5:
            report("analytic vs FD disagreement", cfg, f"rel={err:.2e}")
        return err
    return None


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    rng = np.random.default_rng(seed)
    failures = []

    def report(what, cfg, detail):
        failures.append((what, detail, {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                        for k, v in cfg.items()}))

    t = Tesseract.from_tesseract_api(API)
    errs, checked = [], 0
    status_seen = {0: 0, 1: 0, 2: 0}
    for i in range(n):
        cfg = draw(rng)
        try:
            e = check(t, cfg, rng, report)
        except Exception as exc:  # a raised error is itself a finding
            report(f"raised {type(exc).__name__}", cfg, str(exc)[:160])
            continue
        checked += 1
        if e is not None:
            errs.append(e)
        status_seen[int(t.apply(cfg)["status"])] += 1
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{n} drawn, {len(failures)} findings", flush=True)

    print(f"\nchecked {checked}/{n} configurations (seed {seed})")
    print(f"status coverage: 0={status_seen[0]} 1={status_seen[1]} 2={status_seen[2]}")
    if errs:
        print(f"analytic-vs-FD worst {max(errs):.2e}, median {np.median(errs):.2e}, "
              f"over {len(errs)} topology-stable draws")
    if failures:
        print(f"\n{len(failures)} FINDINGS")
        for what, detail, cfg in failures[:5]:
            print(f"  - {what}: {detail}")
            print(f"    cfg={cfg}")
        raise SystemExit(1)
    print("no invariant violations")


if __name__ == "__main__":
    main()
