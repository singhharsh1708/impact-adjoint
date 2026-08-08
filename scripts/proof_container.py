"""Boundary proof, containerized: jax.grad through the built julia-kernel Docker image."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)


def main():
    # colima only shares $HOME with the VM; the SDK's default /var/folders temp
    # output mount is unwritable inside the container, so pin it under home.
    out_dir = Path(__file__).parent.parent / ".tessout"
    out_dir.mkdir(exist_ok=True)
    with Tesseract.from_image("julia-kernel", output_path=out_dir) as t:
        x = jnp.linspace(-2.0, 2.0, 7, dtype=jnp.float64)

        def loss(x):
            return apply_tesseract(t, {"x": x})["y"].sum()

        val = loss(x)
        grad = jax.grad(loss)(x)

    expected_grad = np.cos(x) * x**2 + 2 * x * np.sin(x)
    np.testing.assert_allclose(float(val), float((np.sin(x) * x**2).sum()), rtol=1e-12)
    np.testing.assert_allclose(np.asarray(grad), expected_grad, rtol=1e-12)
    print("CONTAINER PROOF PASSED: jax.grad through Docker image == analytic (rtol 1e-12)")


if __name__ == "__main__":
    main()
