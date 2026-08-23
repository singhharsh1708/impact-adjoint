from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, Field, model_validator

from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType

from juliacall import Main as jl

jl.include(str(Path(__file__).parent / "julia" / "contact_solver.jl"))

MAX_EVENTS = 8
DIFF_OUTPUTS = {"qf": (0, 4), "impact_x": (4, MAX_EVENTS)}
NROWS = 4 + MAX_EVENTS


def _theta_layout(nb: int) -> dict:
    """θ layout matching contact_solver.jl for an nb-bump terrain.

    Values are (offset, length); scalars have length 0 and are packed and
    unpacked as 0-d.
    """
    return {
        "v0": (0, 2),
        "y0": (2, 0),
        "e": (3, 0),
        "mu": (4, 0),
        "amp": (5, nb),
        "ctr": (5 + nb, nb),
        "wid": (5 + 2 * nb, nb),
    }


class InputSchema(BaseModel):
    v0: Differentiable[Array[(2,), Float64]] = Field(description="Launch velocity (vx, vy). Launch position is (0, y0).")
    y0: Differentiable[Float64] = Field(description="Launch height (must start above the terrain).", default=1.0)
    e: Differentiable[Float64] = Field(description="Normal restitution coefficient in (0, 1].", default=0.7)
    mu: Differentiable[Float64] = Field(description="Tangential velocity loss factor in [0, 1).", default=0.1)
    amp: Differentiable[Array[(None,), Float64]] = Field(
        description="Terrain Gaussian bump amplitudes (any number of bumps; amp/ctr/wid must share one length)."
    )
    ctr: Differentiable[Array[(None,), Float64]] = Field(description="Terrain bump centers.")
    wid: Differentiable[Array[(None,), Float64]] = Field(description="Terrain bump widths (> 0).")
    drag: Float64 = Field(description="Linear drag coefficient, >= 0 (negative drag would add energy without bound). Not differentiated.", default=0.0)
    t_final: Float64 = Field(description="Simulation end time.", default=2.0)
    dt: Float64 = Field(
        description="Integrator step size. Terrain features narrower than |vx|*dt/3 can be stepped over; choose dt <= min(wid)/(3 |vx|).",
        default=1e-3,
    )
    n_samples: int = Field(description="Trajectory sample rows returned for visualization.", default=0)
    v_stop: Float64 = Field(
        description="Normal-velocity floor, >= 0: if the post-impact normal velocity falls below this, the simulation stops at that impact with status=2 (chatter / settled contact; sticking and sliding are outside this model).",
        default=1e-4,
    )

    @model_validator(mode="after")
    def validate_inputs(self) -> Self:
        na, nc, nw = self.amp.shape[0], self.ctr.shape[0], self.wid.shape[0]
        if not (na == nc == nw >= 1):
            raise ValueError(f"amp/ctr/wid must share one length >= 1, got {na}/{nc}/{nw}")
        if self.n_samples < 0:
            raise ValueError(f"n_samples must be >= 0, got {self.n_samples}")
        if isinstance(self.e, ShapeDType):
            # abstract evaluation: shapes only, no values to check. n_samples
            # is checked above rather than below, because it is a plain int and
            # stays concrete here, where a negative value would otherwise reach
            # _abstract_shapes and be published as a negative array dimension.
            return self
        if not (0.0 < float(self.e) <= 1.0):
            raise ValueError(f"e must be in (0, 1], got {float(self.e)} (e > 1 would gain energy)")
        if not (0.0 <= float(self.mu) < 1.0):
            raise ValueError(f"mu must be in [0, 1), got {float(self.mu)}")
        if float(self.drag) < 0.0:
            raise ValueError(
                f"drag must be >= 0, got {float(self.drag)} "
                "(negative drag would add energy without bound)"
            )
        if float(self.v_stop) < 0.0:
            raise ValueError(f"v_stop must be >= 0, got {float(self.v_stop)}")
        if not float(self.dt) > 0.0:
            raise ValueError(f"dt must be > 0, got {float(self.dt)}")
        if not float(self.t_final) >= 0.0:
            raise ValueError(f"t_final must be >= 0, got {float(self.t_final)}")
        if np.any(np.asarray(self.wid) <= 0.0):
            raise ValueError("all terrain widths must be > 0")
        stiffness = float(self.drag) * float(self.dt)
        if stiffness > 2.7:
            raise ValueError(
                f"drag * dt must stay below the explicit RK4 stability limit, "
                f"got {stiffness:.3g} (drag={float(self.drag):.6g}, "
                f"dt={float(self.dt):.6g}). Past roughly 2.785 the fixed step "
                "amplifies instead of damping: energy grows under a purely "
                "dissipative force and the sensitivities change sign, while "
                "the run still reports status 0. Reduce dt or drag."
            )
        for name in ("v0", "y0", "e", "mu", "amp", "ctr", "wid", "drag", "t_final", "dt", "v_stop"):
            if not np.all(np.isfinite(np.asarray(getattr(self, name), dtype=np.float64))):
                raise ValueError(f"{name} must be finite")
        return self


class OutputSchema(BaseModel):
    qf: Differentiable[Array[(4,), Float64]] = Field(
        description="State (x, y, vx, vy) at t_end. For status != 0 this is the state at the truncating event, and its Jacobian is the total derivative including event-time dependence."
    )
    impact_x: Differentiable[Array[(MAX_EVENTS,), Float64]] = Field(
        description="x-coordinate of each impact, zero-padded to MAX_EVENTS. Only entries [:n_events] are meaningful; padded entries are exactly 0 with exactly-zero derivative rows, so gradient checks stay coherent. Always mask by n_events."
    )
    n_events: Int32 = Field(description="Number of impacts.")
    traj: Array[(None, 5), Float64] = Field(description="Sampled (t, x, y, vx, vy) rows for visualization.")
    status: Int32 = Field(
        description="0: ran to t_final. 1: event capacity (MAX_EVENTS) hit; stopped at that crossing. 2: contact settled (post-impact normal velocity < v_stop)."
    )
    t_end: Float64 = Field(description="Time at which qf is measured (== t_final iff status == 0).")


def _pack_theta(inputs: InputSchema, layout: dict) -> np.ndarray:
    theta = np.zeros(5 + 3 * layout["amp"][1])
    for name, (off, ln) in layout.items():
        val = np.asarray(getattr(inputs, name), dtype=np.float64)
        if ln == 0:
            theta[off] = val
        else:
            theta[off : off + ln] = val
    return theta


def _run(inputs: InputSchema, want_sens: bool):
    layout = _theta_layout(int(np.asarray(inputs.amp).shape[0]))
    theta = _pack_theta(inputs, layout)
    qf, Jqf, impact_x, Jimp, nev, traj, status, t_end = jl.ContactSolver.run_solver(
        theta,
        float(inputs.drag),
        float(inputs.t_final),
        float(inputs.dt),
        MAX_EVENTS,
        int(inputs.n_samples),
        float(inputs.v_stop),
        want_sens,
    )
    J = np.vstack([np.array(Jqf, copy=True), np.array(Jimp, copy=True)]) if want_sens else None
    return (
        np.array(qf, dtype=np.float64, copy=True),
        np.array(impact_x, dtype=np.float64, copy=True),
        int(nev),
        np.array(traj, dtype=np.float64, copy=True),
        int(status),
        float(t_end),
        J,
        layout,
    )


def apply(inputs: InputSchema) -> OutputSchema:
    qf, impact_x, nev, traj, status, t_end, _, _ = _run(inputs, want_sens=False)
    return OutputSchema(qf=qf, impact_x=impact_x, n_events=nev, traj=traj, status=status, t_end=t_end)


def _slice_jac(J: np.ndarray, layout: dict, out: str, inp: str) -> np.ndarray:
    row_off, row_len = DIFF_OUTPUTS[out]
    col_off, col_len = layout[inp]
    block = J[row_off : row_off + row_len]
    if col_len == 0:
        return block[:, col_off]
    return block[:, col_off : col_off + col_len]


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    *_, J, layout = _run(inputs, want_sens=True)
    return {out: {inp: _slice_jac(J, layout, out, inp) for inp in jac_inputs} for out in jac_outputs}


def _abstract_shapes(n_samples: int):
    return {
        "qf": ShapeDType(shape=(4,), dtype="float64"),
        "impact_x": ShapeDType(shape=(MAX_EVENTS,), dtype="float64"),
        "n_events": ShapeDType(shape=(), dtype="int32"),
        "traj": ShapeDType(shape=(n_samples, 5), dtype="float64"),
        "status": ShapeDType(shape=(), dtype="int32"),
        "t_end": ShapeDType(shape=(), dtype="float64"),
    }


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict,
):
    *_, J, layout = _run(inputs, want_sens=True)
    theta_dot = np.zeros(J.shape[1])
    for name in jvp_inputs:
        off, ln = layout[name]
        tv = np.asarray(tangent_vector[name], dtype=np.float64)
        if ln == 0:
            theta_dot[off] = tv
        else:
            theta_dot[off : off + ln] = tv
    full = J @ theta_dot
    out = {}
    for name in jvp_outputs:
        off, ln = DIFF_OUTPUTS[name]
        out[name] = full[off : off + ln]
    return out


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    *_, J, layout = _run(inputs, want_sens=True)
    ct = np.zeros(NROWS)
    for name in vjp_outputs:
        off, ln = DIFF_OUTPUTS[name]
        ct[off : off + ln] = np.asarray(cotangent_vector[name], dtype=np.float64)
    full = ct @ J
    out = {}
    for name in vjp_inputs:
        off, ln = layout[name]
        if ln == 0:
            out[name] = full[off]
        else:
            out[name] = full[off : off + ln]
    return out


def abstract_eval(abstract_inputs):
    return _abstract_shapes(int(abstract_inputs.n_samples))
