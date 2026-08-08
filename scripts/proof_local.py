"""Boundary proof, dev mode (no Docker): jax.grad through the Julia kernel Tesseract.

Verifies that gradients flowing through tesseract-jax -> Tesseract VJP endpoint
-> Julia hand-written adjoint match the analytic derivative.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

API_PATH = Path(__file__).parent.parent / "tesseracts" / "julia_kernel" / "tesseract_api.py"


def main():
    t = Tesseract.from_tesseract_api(API_PATH)

    x = jnp.linspace(-2.0, 2.0, 7, dtype=jnp.float64)

    def loss(x):
        res = apply_tesseract(t, {"x": x})
        return res["y"].sum()

    val = loss(x)
    grad = jax.grad(loss)(x)

    expected_y = np.sin(x) * x**2
    expected_grad = np.cos(x) * x**2 + 2 * x * np.sin(x)

    np.testing.assert_allclose(val, expected_y.sum(), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(grad), expected_grad, rtol=1e-12)

    print("apply value :", float(val))
    print("jax.grad    :", np.asarray(grad))
    print("analytic    :", np.asarray(expected_grad))
    print("BOUNDARY PROOF PASSED: jax.grad == Julia analytic adjoint (rtol 1e-12)")


if __name__ == "__main__":
    main()
