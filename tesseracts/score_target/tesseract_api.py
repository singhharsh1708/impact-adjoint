from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from pydantic import BaseModel, Field

jax.config.update("jax_enable_x64", True)

from tesseract_core.runtime import Array, Differentiable, Float64
from tesseract_core.runtime.jax_recipes import (
    jax_abstract_eval,
    jax_apply,
    jax_jacobian,
    jax_jvp,
    jax_vjp,
)


class InputSchema(BaseModel):
    qf: Differentiable[Array[(4,), Float64]] = Field(description="Final state (x, y, vx, vy) from the contact solver.")
    target: Differentiable[Array[(2,), Float64]] = Field(description="Cup position (x*, y*).")
    weights: Array[(3,), Float64] = Field(
        description="Loss weights (position-x, position-y, kinetic).",
    )


class OutputSchema(BaseModel):
    loss: Differentiable[Float64] = Field(
        description="w1 (x - x*)^2 + w2 (y - y*)^2 + w3 |v|^2. Land in the cup, gently."
    )
    miss_distance: Differentiable[Float64] = Field(description="Euclidean distance from (x, y) to the cup.")


@eqx.filter_jit
def apply_jit(inputs: dict) -> dict:
    qf = inputs["qf"]
    target = inputs["target"]
    w = inputs["weights"]
    dx = qf[0] - target[0]
    dy = qf[1] - target[1]
    ke = qf[2] ** 2 + qf[3] ** 2
    return {
        "loss": w[0] * dx**2 + w[1] * dy**2 + w[2] * ke,
        # eps keeps the sqrt differentiable at an exact hit (grad of sqrt(0) is nan)
        "miss_distance": jnp.sqrt(dx**2 + dy**2 + 1e-30),
    }


def apply(inputs: InputSchema) -> OutputSchema:
    return OutputSchema(**jax_apply(apply_jit, inputs))


def jacobian(inputs: InputSchema, jac_inputs: set[str], jac_outputs: set[str]):
    return jax_jacobian(apply_jit, inputs, jac_inputs, jac_outputs)


def jacobian_vector_product(
    inputs: InputSchema,
    jvp_inputs: set[str],
    jvp_outputs: set[str],
    tangent_vector: dict[str, Any],
):
    return jax_jvp(apply_jit, inputs, jvp_inputs, jvp_outputs, tangent_vector)


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict[str, Any],
):
    return jax_vjp(apply_jit, inputs, vjp_inputs, vjp_outputs, cotangent_vector)


def abstract_eval(abstract_inputs):
    return jax_abstract_eval(apply_jit, abstract_inputs)
