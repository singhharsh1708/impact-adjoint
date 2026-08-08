"""Golden regression tests for the contact-sim Tesseract (dev mode, no Docker).

Golden values were produced by the validated solver (see scripts/validate_*.py
for the independent oracles that justify them) and pin behavior against
refactors: trajectory primals, analytic Jacobian entries, termination
semantics, and energy conservation.
"""

from pathlib import Path

import numpy as np
import pytest

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add("ForwardDiff")')

from tesseract_core import Tesseract

API_PATH = Path(__file__).parent.parent / "tesseracts" / "contact_sim" / "tesseract_api.py"

BASE = {
    "v0": np.array([2.0, 0.5]),
    "y0": 1.0,
    "e": 0.7,
    "mu": 0.1,
    "amp": np.array([0.2, 0.1, 0.15]),
    "ctr": np.array([1.0, 2.5, 4.0]),
    "wid": np.array([0.5, 0.4, 0.6]),
    "drag": 0.0,
    "t_final": 2.0,
    "dt": 1e-3,
    "n_samples": 0,
}


@pytest.fixture(scope="session")
def tess():
    return Tesseract.from_tesseract_api(API_PATH)


def test_base_trajectory(tess):
    r = tess.apply(BASE)
    assert int(r["n_events"]) == 4
    assert int(r["status"]) == 0
    np.testing.assert_allclose(
        np.asarray(r["qf"]), [3.176208, 0.108420, 1.432017, -0.699560], atol=2e-6
    )
    np.testing.assert_allclose(
        np.asarray(r["impact_x"])[:4], [0.917391, 1.800960, 2.494558, 2.934206], atol=2e-6
    )


def test_base_jacobian_entries(tess):
    j = tess.jacobian(BASE, jac_inputs={"e", "v0"}, jac_outputs={"impact_x", "qf"})
    np.testing.assert_allclose(
        np.asarray(j["impact_x"]["e"])[:4], [0.0, 0.941025, 2.536210, 4.472073], atol=2e-5
    )
    # first impact is independent of e; d impact_1/d v0x = time of first impact
    assert abs(float(np.asarray(j["impact_x"]["e"])[0])) < 1e-12
    assert abs(float(np.asarray(j["impact_x"]["v0"])[0, 0]) - 0.444130) < 2e-6


def test_flat_terrain_closed_form(tess):
    flat = dict(BASE, amp=np.zeros(3), t_final=0.3)
    r = tess.apply(flat)
    T = 0.3
    expect = [2.0 * T, 1.0 + 0.5 * T - 0.5 * 9.81 * T**2, 2.0, 0.5 - 9.81 * T]
    np.testing.assert_allclose(np.asarray(r["qf"]), expect, rtol=1e-12)


def test_settle_termination(tess):
    r = tess.apply(dict(BASE, e=0.05, n_samples=500))
    assert int(r["status"]) == 2
    assert float(r["t_end"]) < BASE["t_final"]


def test_event_capacity_termination(tess):
    r = tess.apply(dict(BASE, e=0.85, mu=0.0, t_final=6.0, n_samples=500))
    assert int(r["status"]) == 1
    assert int(r["n_events"]) == 8
    traj = np.asarray(r["traj"])
    amp, ctr, wid = BASE["amp"], BASE["ctr"], BASE["wid"]
    h = np.sum(amp * np.exp(-((traj[:, 1:2] - ctr) ** 2) / (2 * wid**2)), axis=1)
    assert float(np.min(traj[:, 2] - h)) > -1e-9


def test_energy_conservation(tess):
    r = tess.apply(dict(BASE, e=1.0, mu=0.0, t_final=5.0, n_samples=2000))
    traj = np.asarray(r["traj"])
    E = 0.5 * (traj[:, 3] ** 2 + traj[:, 4] ** 2) + 9.81 * traj[:, 2]
    assert float(np.max(np.abs(E - E[0])) / E[0]) < 1e-11


def test_vjp_matches_jacobian(tess):
    j = tess.jacobian(BASE, jac_inputs={"e"}, jac_outputs={"qf"})
    v = tess.vector_jacobian_product(
        BASE, vjp_inputs={"e"}, vjp_outputs={"qf"},
        cotangent_vector={"qf": np.array([1.0, 0.0, 0.0, 0.0])},
    )
    assert abs(float(np.asarray(v["e"])) - float(np.asarray(j["qf"]["e"])[0])) < 1e-12
