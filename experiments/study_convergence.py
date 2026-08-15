"""Convergence study: does the solver converge at the expected order?

Two things are measured, both against the flat-terrain closed form that
scripts/validate_closed_form.py derives symbolically (so the reference does
not come from this solver):

1. Trajectory error vs dt on a multi-bounce run. RK4 is exact on the
   ballistic arcs here, so the error floor is set by event localization
   (80 bisection steps), not by the integrator order.
2. A smooth pre-impact arc with linear drag, against the analytic solution of
   the drag ODE. No events fire, so this isolates the integrator and the
   observed slope is the empirical order of accuracy.

Writes experiments/convergence_result.npz.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

from tesseract_core import Tesseract

ROOT = Path(__file__).parent.parent
API = ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py"

G = 9.81
BASE = {
    "v0": np.array([2.0, 0.5]), "y0": 1.0, "e": 0.7, "mu": 0.1,
    "amp": np.zeros(3), "ctr": np.array([1.0, 2.5, 4.0]),
    "wid": np.array([0.5, 0.4, 0.6]),
    "drag": 0.0, "t_final": 2.0, "n_samples": 0,
}
DTS = np.array([2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4])


def closed_form_flat(v0, y0, e, mu, t_final):
    """Exact multi-bounce solution on flat terrain with no drag."""
    vx0, vy0 = float(v0[0]), float(v0[1])
    w0 = np.sqrt(vy0**2 + 2 * G * y0)
    t_imp = (vy0 + w0) / G
    x_imp = vx0 * t_imp
    xs, ts = [x_imp], [t_imp]
    vx, vy = (1 - mu) * vx0, e * w0
    while True:
        flight = 2 * vy / G
        if ts[-1] + flight > t_final:
            break
        ts.append(ts[-1] + flight)
        xs.append(xs[-1] + vx * flight)
        vx, vy = (1 - mu) * vx, e * vy
    dt_rem = t_final - ts[-1]
    qf = np.array([xs[-1] + vx * dt_rem, vy * dt_rem - 0.5 * G * dt_rem**2,
                   vx, vy - G * dt_rem])
    return qf, np.array(xs)


def analytic_drag(v0, y0, cd, T):
    """Exact solution of dv/dt = -cd v - g e_y over a smooth arc."""
    vx0, vy0 = float(v0[0]), float(v0[1])
    ex = np.exp(-cd * T)
    vx = vx0 * ex
    x = vx0 / cd * (1 - ex)
    vy = (vy0 + G / cd) * ex - G / cd
    y = y0 + (vy0 + G / cd) / cd * (1 - ex) - G * T / cd
    return np.array([x, y, vx, vy])


def sweep(t, cfg, ref_qf):
    errs = []
    for dt in DTS:
        r = t.apply({**cfg, "dt": float(dt)})
        errs.append(float(np.max(np.abs(np.asarray(r["qf"]) - ref_qf))))
    return np.array(errs)


def observed_order(dts, errs):
    """Slope of log(err) vs log(dt), fitted only where the error is still
    above the roundoff/event-localization floor. Points at the floor would
    flatten the fit and understate the order."""
    floor = np.min(errs[errs > 0])
    keep = errs > 100 * floor
    if keep.sum() < 2:
        return float("nan"), keep
    slope = float(np.polyfit(np.log(dts[keep]), np.log(errs[keep]), 1)[0])
    return slope, keep


def main():
    t = Tesseract.from_tesseract_api(API)

    ref_qf, ref_imp = closed_form_flat(BASE["v0"], BASE["y0"], BASE["e"], BASE["mu"], BASE["t_final"])
    errs_flat = sweep(t, BASE, ref_qf)
    print("flat terrain, no drag (RK4 exact on arcs; floor = event localization)")
    for dt, e in zip(DTS, errs_flat):
        print(f"  dt={dt:8.1e}  max|qf - closed form| = {e:.3e}")

    # smooth arc with drag, no events: isolates the integrator so the RK4
    # order is what is actually being measured.
    CD, T_SMOOTH = 0.4, 0.35
    drag_cfg = {**BASE, "drag": CD, "t_final": T_SMOOTH}
    ref_smooth = analytic_drag(BASE["v0"], BASE["y0"], CD, T_SMOOTH)
    assert int(t.apply({**drag_cfg, "dt": 1e-3})["n_events"]) == 0, "arc must stay smooth"
    errs_drag = sweep(t, drag_cfg, ref_smooth)
    print("\nsmooth arc, drag 0.4, no events (vs analytic drag solution)")
    for dt, e in zip(DTS, errs_drag):
        print(f"  dt={dt:8.1e}  max|qf - analytic| = {e:.3e}")

    ord_drag, keep = observed_order(DTS, errs_drag)
    print(f"\nfitted over dt = {DTS[keep]} (points above the error floor)")
    print(f"observed order of accuracy (smooth arc): {ord_drag:.2f}")

    np.savez(ROOT / "experiments" / "convergence_result.npz",
             dts=DTS, err_flat=errs_flat, err_drag=errs_drag,
             order_drag=ord_drag, fit_mask=keep, ref_qf=ref_qf, ref_impacts=ref_imp)

    assert errs_flat[-1] < 1e-9, f"flat-terrain floor too high: {errs_flat[-1]:.2e}"
    assert 3.5 < ord_drag < 4.5, f"expected 4th order on a smooth arc, got {ord_drag:.2f}"
    print("\nCONVERGENCE STUDY PASSED: flat-terrain error at machine-precision floor, "
          f"smooth arc converges at order {ord_drag:.2f}")


if __name__ == "__main__":
    main()
