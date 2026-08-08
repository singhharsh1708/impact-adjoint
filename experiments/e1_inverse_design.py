"""E1: inverse design through two composed Tesseracts.

Pipeline: (v0, e) -> [contact-sim: Julia, saltation sensitivities] -> qf
          -> [score-target: JAX autodiff] -> loss.
One jax.grad spans both Tesseracts via tesseract-jax; optax Adam finds launch
velocity and restitution that land the ball in a cup on bumpy terrain after
multiple bounces.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add("ForwardDiff")')

import jax
import jax.numpy as jnp
import optax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parent.parent
TERRAIN = {
    "amp": np.array([0.2, 0.1, 0.15]),
    "ctr": np.array([1.0, 2.5, 4.0]),
    "wid": np.array([0.5, 0.4, 0.6]),
}


def terrain_h(x):
    return float(np.sum(TERRAIN["amp"] * np.exp(-((x - TERRAIN["ctr"]) ** 2) / (2 * TERRAIN["wid"] ** 2))))


TARGET_X = 4.3
TARGET = np.array([TARGET_X, terrain_h(TARGET_X)])
WEIGHTS = np.array([1.0, 1.0, 0.005])


def main():
    sim = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    score = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "score_target" / "tesseract_api.py")

    def loss_fn(params):
        res = apply_tesseract(
            sim,
            {
                "v0": params["v0"], "y0": 1.0, "e": params["e"], "mu": 0.1,
                "amp": TERRAIN["amp"], "ctr": TERRAIN["ctr"], "wid": TERRAIN["wid"],
                "drag": 0.0, "t_final": 2.0, "dt": 1e-3, "n_samples": 0, "v_stop": 1e-4,
            },
        )
        sc = apply_tesseract(score, {"qf": res["qf"], "target": jnp.asarray(TARGET), "weights": jnp.asarray(WEIGHTS)})
        return sc["loss"], (sc["miss_distance"], res["n_events"])

    params = {"v0": jnp.array([2.0, 0.5]), "e": jnp.asarray(0.7)}
    opt = optax.adam(learning_rate=0.03)
    state = opt.init(params)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    n_iters = 150
    history = []
    for it in range(n_iters):
        (loss, (miss, nev)), grads = grad_fn(params)
        history.append((it, float(loss), float(miss), int(nev)))
        if it % 15 == 0 or it == n_iters - 1:
            print(f"iter {it:3d}  loss {float(loss):10.6f}  miss {float(miss):8.5f} m  bounces {int(nev)}")
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
        params["e"] = jnp.clip(params["e"], 0.05, 0.95)

    final_loss, final_miss = history[-1][1], history[-1][2]
    first_loss, first_miss = history[0][1], history[0][2]
    print(f"\noptimized v0 = {np.asarray(params['v0']).round(5)}, e = {float(params['e']):.5f}")
    print(f"loss {first_loss:.4f} -> {final_loss:.6f}, miss {first_miss:.3f} m -> {final_miss:.4f} m, bounces {history[-1][3]}")
    assert final_miss < 0.05, f"did not land in cup: miss {final_miss:.4f} m"
    assert history[-1][3] >= 2, "want a multi-bounce solution"
    np.save(ROOT / "experiments" / "e1_history.npy", np.array([(h[1], h[2]) for h in history]))
    print("E1 PASSED: gradient descent through both Tesseracts lands the ball in the cup")


if __name__ == "__main__":
    main()
