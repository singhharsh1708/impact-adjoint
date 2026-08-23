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

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

from tesseract_core import Tesseract
from tesseract_core.runtime.core import load_module_from_path

API_PATH = Path(__file__).parent.parent / "tesseracts" / "contact_sim" / "tesseract_api.py"

# import the module once so the tests reference the API's own constants rather
# than duplicating them (and so Julia is included exactly once)
_api = load_module_from_path(API_PATH)
MAX_EVENTS = _api.MAX_EVENTS

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
    return Tesseract.from_tesseract_api(_api)


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
    assert int(r["n_events"]) == MAX_EVENTS
    traj = np.asarray(r["traj"])
    amp, ctr, wid = BASE["amp"], BASE["ctr"], BASE["wid"]
    h = np.sum(amp * np.exp(-((traj[:, 1:2] - ctr) ** 2) / (2 * wid**2)), axis=1)
    assert float(np.min(traj[:, 2] - h)) > -1e-9


def test_energy_conservation(tess):
    r = tess.apply(dict(BASE, e=1.0, mu=0.0, t_final=5.0, n_samples=2000))
    traj = np.asarray(r["traj"])
    E = 0.5 * (traj[:, 3] ** 2 + traj[:, 4] ** 2) + 9.81 * traj[:, 2]
    assert float(np.max(np.abs(E - E[0])) / E[0]) < 1e-11


def _fd_qf(tess, cfg, name, idx, h=1e-6):
    def shift(sign):
        c = {k: (np.array(v, dtype=float) if isinstance(v, np.ndarray) else v) for k, v in cfg.items()}
        val = np.atleast_1d(np.array(c[name], dtype=float)).copy()
        val[idx] += sign * h
        c[name] = val if val.size > 1 else float(val[0])
        return np.asarray(tess.apply(c)["qf"])

    return (shift(+1) - shift(-1)) / (2 * h)


def test_five_bump_jacobian_vs_fd(tess):
    # pins the dynamic theta-layout index math at a bump count never used elsewhere
    cfg = dict(BASE, amp=np.array([0.1, 0.05, 0.12, 0.07, 0.09]),
               ctr=np.array([0.8, 1.6, 2.4, 3.2, 4.0]), wid=np.full(5, 0.4))
    r = tess.apply(cfg)
    assert int(r["status"]) == 0 and int(r["n_events"]) >= 2
    j = tess.jacobian(cfg, jac_inputs={"amp", "e"}, jac_outputs={"qf"})
    J = np.asarray(j["qf"]["amp"])
    assert J.shape == (4, 5)
    for i in (0, 2, 4):
        fd = _fd_qf(tess, cfg, "amp", i)
        np.testing.assert_allclose(J[:, i], fd, rtol=1e-4, atol=1e-7)


def test_truncation_total_derivatives_vs_fd(tess):
    # status=2 (settled) and status=1 (event capacity) both report total
    # derivatives including event-time dependence -- the subtlest solver code
    settle = dict(BASE, e=0.05)
    r = tess.apply(settle)
    assert int(r["status"]) == 2
    J = np.asarray(tess.jacobian(settle, jac_inputs={"y0"}, jac_outputs={"qf"})["qf"]["y0"])
    fd = _fd_qf(tess, settle, "y0", 0)
    np.testing.assert_allclose(J, fd, rtol=1e-4, atol=1e-6)

    cap = dict(BASE, e=0.85, mu=0.0, t_final=6.0)
    r = tess.apply(cap)
    assert int(r["status"]) == 1
    J = np.asarray(tess.jacobian(cap, jac_inputs={"e"}, jac_outputs={"qf"})["qf"]["e"])
    fd = _fd_qf(tess, cap, "e", 0)
    np.testing.assert_allclose(J, fd, rtol=1e-4, atol=1e-6)


def test_composed_vjp_chain(tess):
    # contact_sim -> score_target under one jax.grad matches FD of the chain
    import jax
    import jax.numpy as jnp
    from tesseract_jax import apply_tesseract

    jax.config.update("jax_enable_x64", True)
    score = Tesseract.from_tesseract_api(Path(__file__).parent.parent / "tesseracts" / "score_target" / "tesseract_api.py")

    def loss(e):
        res = apply_tesseract(tess, {**BASE, "e": e})
        sc = apply_tesseract(score, {"qf": res["qf"], "target": jnp.array([3.0, 0.1]),
                                     "weights": jnp.array([1.0, 1.0, 0.01])})
        return sc["loss"]

    g = float(jax.grad(loss)(jnp.asarray(0.7)))
    h = 1e-6
    fd = (float(loss(jnp.asarray(0.7 + h))) - float(loss(jnp.asarray(0.7 - h)))) / (2 * h)
    assert abs(g - fd) / (abs(fd) + 1.0) < 1e-5


def test_input_bounds_rejected(tess):
    cases = [
        (dict(BASE, e=1.3), "e must be in"),
        (dict(BASE, mu=-0.2), "mu must be in"),
        (dict(BASE, dt=0.0), "dt must be > 0"),
        (dict(BASE, wid=np.array([0.5, -0.1, 0.6])), "widths must be > 0"),
        (dict(BASE, drag=-1.0), "drag must be >= 0"),
        (dict(BASE, v_stop=-1.0), "v_stop must be >= 0"),
        (dict(BASE, t_final=-1.0), "t_final must be >= 0"),
        (dict(BASE, n_samples=-3), "n_samples must be >= 0"),
        (dict(BASE, ctr=np.array([1.0, 2.5])), "must share one length"),
        (dict(BASE, drag=float("nan")), "must be finite"),
        (dict(BASE, drag=1e6), "stability limit"),
    ]
    for bad, expected in cases:
        with pytest.raises(Exception) as excinfo:
            tess.apply(bad)
        assert expected in str(excinfo.value), (
            f"expected a message containing {expected!r}, got {excinfo.value}"
        )


def test_jvp_matches_jacobian(tess):
    # the jvp endpoint's scalar-tangent packing branch is exercised nowhere else
    j = tess.jacobian(BASE, jac_inputs={"v0", "e"}, jac_outputs={"qf"})
    tv = {"v0": np.array([0.3, -0.2]), "e": 0.5}
    out = tess.jacobian_vector_product(
        BASE, jvp_inputs={"v0", "e"}, jvp_outputs={"qf"}, tangent_vector=tv
    )
    expect = np.asarray(j["qf"]["v0"]) @ tv["v0"] + np.asarray(j["qf"]["e"]) * tv["e"]
    np.testing.assert_allclose(np.asarray(out["qf"]), expect, rtol=1e-12, atol=1e-12)


def test_vjp_matches_jacobian(tess):
    # A one-hot cotangent on the first row makes ct @ J identical to J[0], so
    # an endpoint that ignored the cotangent and returned the first row would
    # pass. Every component is distinct and nonzero here.
    ct = np.array([1.0, -2.0, 0.5, 3.0])
    j = tess.jacobian(BASE, jac_inputs={"e"}, jac_outputs={"qf"})
    v = tess.vector_jacobian_product(
        BASE, vjp_inputs={"e"}, vjp_outputs={"qf"},
        cotangent_vector={"qf": ct},
    )
    expect = float(ct @ np.asarray(j["qf"]["e"], float))
    assert abs(float(np.asarray(v["e"])) - expect) < 1e-12

def test_large_bump_count_jacobian(tess):
    """nb=24 (77 theta) is the E5 configuration and the only one that exercises
    ForwardDiff's chunked path; nothing else in the suite reaches it."""
    nb = 24
    cfg = dict(
        BASE,
        amp=np.full(nb, 0.05),
        ctr=np.linspace(0.4, 5.4, nb),
        wid=np.full(nb, 0.18),
    )
    jac = tess.jacobian(cfg, jac_inputs={"amp"}, jac_outputs={"qf"})
    a = np.asarray(jac["qf"]["amp"], dtype=float)
    assert a.shape == (4, nb)
    assert np.all(np.isfinite(a))

    # central difference on one interior amplitude, at fixed event topology
    k, h = nb // 2, 1e-6
    up, dn = np.array(cfg["amp"]), np.array(cfg["amp"])
    up[k] += h
    dn[k] -= h
    ru, rd = tess.apply({**cfg, "amp": up}), tess.apply({**cfg, "amp": dn})
    assert int(ru["n_events"]) == int(rd["n_events"]) == int(tess.apply(cfg)["n_events"]), (
        "the probe crossed a bounce-count boundary, so this test compared "
        "nothing; pick a smaller h or a different configuration"
    )
    fd = (np.asarray(ru["qf"], float) - np.asarray(rd["qf"], float)) / (2 * h)
    assert np.max(np.abs(a[:, k] - fd)) / max(1.0, np.max(np.abs(fd))) < 2e-5


def test_impact_x_padding_is_exactly_zero(tess):
    """The padded rows of impact_x and their derivatives are documented as
    exactly zero; nothing verified it."""
    r = tess.apply(BASE)
    nev = int(r["n_events"])
    imp = np.asarray(r["impact_x"], dtype=float)
    assert nev < len(imp), "config should not saturate the event budget"
    assert np.all(imp[nev:] == 0.0)

    jac = tess.jacobian(BASE, jac_inputs={"v0"}, jac_outputs={"impact_x"})
    j = np.asarray(jac["impact_x"]["v0"], dtype=float)
    assert np.all(j[nev:, :] == 0.0), "padded derivative rows must be exactly zero"


def test_drag_is_differentiated_consistently(tess):
    """No golden test used drag != 0, so the drag sector of the Jacobian was
    exercised only by the reference oracle."""
    cfg = dict(BASE, drag=0.3)
    r = tess.apply(cfg)
    assert int(r["status"]) == 0
    jac = tess.jacobian(cfg, jac_inputs={"v0"}, jac_outputs={"qf"})
    a = np.asarray(jac["qf"]["v0"], dtype=float)
    assert np.all(np.isfinite(a))
    h = 1e-6
    up = tess.apply({**cfg, "v0": np.asarray(cfg["v0"]) + np.array([0.0, h])})
    dn = tess.apply({**cfg, "v0": np.asarray(cfg["v0"]) - np.array([0.0, h])})
    assert int(up["n_events"]) == int(dn["n_events"]) == int(r["n_events"]), (
        "the probe crossed a bounce-count boundary, so this test compared "
        "nothing; pick a smaller h or a different configuration"
    )
    fd = (np.asarray(up["qf"], float) - np.asarray(dn["qf"], float)) / (2 * h)
    assert np.max(np.abs(a[:, 1] - fd)) / max(1.0, np.max(np.abs(fd))) < 2e-5


def test_rk4_stability_bound_is_pinned_to_the_real_limit(tess):
    """The drag*dt bound sits at RK4's stability boundary, not near it.

    Replacing the threshold with 999.0 or with 0.001 both left the suite
    green: the only case exercising it used a product of 1000, which pins the
    constant to six orders of free play. These straddle it.
    """
    from tesseract_core.runtime.core import load_module_from_path

    api = load_module_from_path(str(API_PATH))
    limit = api.RK4_STABILITY_LIMIT
    assert abs(limit - 2.785293563405281) < 1e-12, (
        f"RK4_STABILITY_LIMIT is {limit}, not the real root of "
        "z^3 + 4z^2 + 12z + 24"
    )

    dt = 1e-3
    tess.apply(dict(BASE, drag=(limit - 1e-6) / dt, dt=dt, t_final=0.05))

    with pytest.raises(Exception) as excinfo:
        tess.apply(dict(BASE, drag=(limit + 1e-6) / dt, dt=dt, t_final=0.05))
    assert "stability limit" in str(excinfo.value)


def test_non_finite_inputs_report_finiteness_not_stability(tess):
    """An infinite drag is an infinite drag, not a stability-limit violation."""
    for bad in (dict(BASE, drag=float("inf")), dict(BASE, dt=float("inf"))):
        with pytest.raises(Exception) as excinfo:
            tess.apply(bad)
        assert "must be finite" in str(excinfo.value), (
            f"expected a finiteness error, got {excinfo.value}"
        )
