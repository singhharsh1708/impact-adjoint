from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType

from juliacall import Main as jl

jl.include(str(Path(__file__).parent / "julia" / "contact_solver.jl"))

MAX_EVENTS = 8
NB = 3
NTH = 5 + 3 * NB

# θ layout must match contact_solver.jl. Values are (offset, length); scalars
# have length 0 and are packed/unpacked as 0-d.
THETA_LAYOUT = {
    "v0": (0, 2),
    "y0": (2, 0),
    "e": (3, 0),
    "mu": (4, 0),
    "amp": (5, NB),
    "ctr": (5 + NB, NB),
    "wid": (5 + 2 * NB, NB),
}
DIFF_OUTPUTS = {"qf": (0, 4), "impact_x": (4, MAX_EVENTS)}
NROWS = 4 + MAX_EVENTS


class InputSchema(BaseModel):
    v0: Differentiable[Array[(2,), Float64]] = Field(description="Launch velocity (vx, vy). Launch position is (0, y0).")
    y0: Differentiable[Float64] = Field(description="Launch height.", default=1.0)
    e: Differentiable[Float64] = Field(description="Normal restitution coefficient in (0, 1].", default=0.7)
    mu: Differentiable[Float64] = Field(description="Tangential velocity loss factor in [0, 1).", default=0.1)
    amp: Differentiable[Array[(NB,), Float64]] = Field(description="Terrain Gaussian bump amplitudes.")
    ctr: Differentiable[Array[(NB,), Float64]] = Field(description="Terrain bump centers.")
    wid: Differentiable[Array[(NB,), Float64]] = Field(description="Terrain bump widths (> 0).")
    drag: Float64 = Field(description="Linear drag coefficient (not differentiated).", default=0.0)
    t_final: Float64 = Field(description="Simulation end time.", default=2.0)
    dt: Float64 = Field(
        description="Integrator step size. Terrain features narrower than |vx|*dt/3 can be stepped over; choose dt <= min(wid)/(3 |vx|).",
        default=1e-3,
    )
    n_samples: int = Field(description="Trajectory sample rows returned for visualization.", default=0)
    v_stop: Float64 = Field(
        description="Normal-velocity floor: if the post-impact normal velocity falls below this, the simulation stops at that impact with status=2 (chatter / settled contact; sticking and sliding are outside this model).",
        default=1e-4,
    )


class OutputSchema(BaseModel):
    qf: Differentiable[Array[(4,), Float64]] = Field(
        description="State (x, y, vx, vy) at t_end. For status != 0 this is the state at the truncating event, and its Jacobian is the total derivative including event-time dependence."
    )
    impact_x: Differentiable[Array[(MAX_EVENTS,), Float64]] = Field(
        description="x-coordinate of each impact, NaN-padded to MAX_EVENTS. Convention: padded entries are NaN in the value but their Jacobian/VJP/JVP rows are exactly zero — only entries [:n_events] are meaningful."
    )
    n_events: Int32 = Field(description="Number of impacts.")
    traj: Array[(None, 5), Float64] = Field(description="Sampled (t, x, y, vx, vy) rows for visualization.")
    status: Int32 = Field(
        description="0: ran to t_final. 1: event capacity (MAX_EVENTS) hit; stopped at that crossing. 2: contact settled (post-impact normal velocity < v_stop)."
    )
    t_end: Float64 = Field(description="Time at which qf is measured (== t_final iff status == 0).")


def _pack_theta(inputs: InputSchema) -> np.ndarray:
    theta = np.zeros(NTH)
    for name, (off, ln) in THETA_LAYOUT.items():
        val = np.asarray(getattr(inputs, name), dtype=np.float64)
        if ln == 0:
            theta[off] = val
        else:
            theta[off : off + ln] = val
    return theta


def _run(inputs: InputSchema, want_sens: bool):
    theta = _pack_theta(inputs)
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
    )


def apply(inputs: InputSchema) -> OutputSchema:
    qf, impact_x, nev, traj, status, t_end, _ = _run(inputs, want_sens=False)
    return OutputSchema(qf=qf, impact_x=impact_x, n_events=nev, traj=traj, status=status, t_end=t_end)


def _slice_jac(J: np.ndarray, out: str, inp: str) -> np.ndarray:
    row_off, row_len = DIFF_OUTPUTS[out]
    col_off, col_len = THETA_LAYOUT[inp]
    block = J[row_off : row_off + row_len]
    if col_len == 0:
        return block[:, col_off]
    return block[:, col_off : col_off + col_len]


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    *_, J = _run(inputs, want_sens=True)
    return {out: {inp: _slice_jac(J, out, inp) for inp in jac_inputs} for out in jac_outputs}


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
    *_, J = _run(inputs, want_sens=True)
    theta_dot = np.zeros(NTH)
    for name in jvp_inputs:
        off, ln = THETA_LAYOUT[name]
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
    *_, J = _run(inputs, want_sens=True)
    ct = np.zeros(NROWS)
    for name in vjp_outputs:
        off, ln = DIFF_OUTPUTS[name]
        ct[off : off + ln] = np.asarray(cotangent_vector[name], dtype=np.float64)
    full = ct @ J
    out = {}
    for name in vjp_inputs:
        off, ln = THETA_LAYOUT[name]
        if ln == 0:
            out[name] = full[off]
        else:
            out[name] = full[off : off + ln]
    return out


def abstract_eval(abstract_inputs):
    return _abstract_shapes(int(abstract_inputs.n_samples))
