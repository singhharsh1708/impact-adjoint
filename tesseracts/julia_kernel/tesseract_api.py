from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

# Load the Julia kernel once per process. juliacall keeps a persistent Julia
# runtime, so repeated endpoint calls do not pay startup cost.
from juliacall import Main as jl

jl.include(str(Path(__file__).parent / "julia" / "kernel.jl"))


class InputSchema(BaseModel):
    x: Differentiable[Array[(None,), Float64]] = Field(
        description="Input vector passed to the Julia kernel."
    )


class OutputSchema(BaseModel):
    y: Differentiable[Array[(None,), Float64]] = Field(
        description="Kernel output sin(x) * x^2, computed in Julia."
    )


def apply(inputs: InputSchema) -> OutputSchema:
    y = np.asarray(jl.Kernel.apply_kernel(np.asarray(inputs.x, dtype=np.float64)))
    return OutputSchema(y=y)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    assert vjp_inputs == {"x"} and vjp_outputs == {"y"}
    x = np.asarray(inputs.x, dtype=np.float64)
    ct = np.asarray(cotangent_vector["y"], dtype=np.float64)
    return {"x": np.asarray(jl.Kernel.vjp_kernel(x, ct))}


def abstract_eval(abstract_inputs):
    return {"y": ShapeDType(shape=abstract_inputs.x.shape, dtype="float64")}
