"""E4: passive sorter. Terrain shape design through contact events.

Design problem: one fixed terrain must passively route balls entering at two
different speeds to two different cups. The nine terrain parameters (bump
amplitudes, centers, widths) are the design variables; gradients flow from
both trajectories' landing errors back through every impact into the shape.

This is shape optimization in the Track-1 sense: the optimized artifact is a
passive structure, and the objective is only computable through the events the
structure itself creates. A vibratory part feeder or a passive chute sorter is
the industrial version of this problem.
"""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

import jax
import jax.numpy as jnp
import optax
from tesseract_core import Tesseract
from tesseract_jax import apply_tesseract

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).parent.parent

FIXED = {
    "y0": 1.0, "e": 0.7, "mu": 0.1,
    "drag": 0.0, "t_final": 2.2, "dt": 1e-3, "n_samples": 0, "v_stop": 1e-4,
}
V_SLOW = np.array([1.6, 0.3])
V_FAST = np.array([2.6, 0.3])
CUP_SLOW = 2.6
CUP_FAST = 4.6
KE_W = 0.005

INIT = {
    "amp": jnp.array([0.15, 0.15, 0.15]),
    "ctr": jnp.array([1.2, 2.4, 3.6]),
    "wid": jnp.array([0.5, 0.5, 0.5]),
}


def main():
    sim = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    score = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "score_target" / "tesseract_api.py")

    def terrain_h_jnp(x, p):
        return jnp.sum(p["amp"] * jnp.exp(-((x - p["ctr"]) ** 2) / (2 * p["wid"] ** 2)))

    def one_ball(p, v0, cup_x):
        res = apply_tesseract(sim, {**FIXED, "v0": v0, **p})
        target = jnp.stack([cup_x, terrain_h_jnp(cup_x, p)])
        sc = apply_tesseract(score, {"qf": res["qf"], "target": target,
                                     "weights": jnp.array([1.0, 1.0, KE_W])})
        return sc["loss"], sc["miss_distance"], res["n_events"]

    def loss_fn(p):
        l1, m1, n1 = one_ball(p, jnp.asarray(V_SLOW), CUP_SLOW)
        l2, m2, n2 = one_ball(p, jnp.asarray(V_FAST), CUP_FAST)
        return l1 + l2, (m1, m2, n1, n2)

    params = dict(INIT)
    opt = optax.adam(learning_rate=0.02)
    state = opt.init(params)
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    n_iters = 200
    hist = []
    for it in range(n_iters):
        (loss, (m1, m2, n1, n2)), grads = grad_fn(params)
        hist.append([float(loss), float(m1), float(m2)])
        evaluated = {k: np.asarray(v) for k, v in params.items()}
        if it % 20 == 0 or it == n_iters - 1:
            print(f"iter {it:3d}  loss {float(loss):9.5f}  miss_slow {float(m1):7.4f}  "
                  f"miss_fast {float(m2):7.4f}  bounces {int(n1)}/{int(n2)}")
        updates, state = opt.update(grads, state)
        params = optax.apply_updates(params, updates)
        params["amp"] = jnp.clip(params["amp"], 0.0, 0.45)
        params["wid"] = jnp.clip(params["wid"], 0.25, 1.2)
        params["ctr"] = jnp.clip(params["ctr"], 0.5, 5.0)

    m1, m2 = hist[-1][1], hist[-1][2]
    print(f"\ndesigned terrain: amp={evaluated['amp'].round(4)}")
    print(f"                  ctr={evaluated['ctr'].round(4)}")
    print(f"                  wid={evaluated['wid'].round(4)}")
    print(f"slow ball -> cup at {CUP_SLOW}: miss {hist[0][1]:.3f} -> {m1:.4f} m")
    print(f"fast ball -> cup at {CUP_FAST}: miss {hist[0][2]:.3f} -> {m2:.4f} m")
    np.savez(ROOT / "experiments" / "e4_result.npz",
             hist=np.array(hist),
             amp=evaluated["amp"], ctr=evaluated["ctr"], wid=evaluated["wid"])
    assert m1 < 0.08 and m2 < 0.08, f"sorter did not converge: {m1:.3f}, {m2:.3f}"
    print("E4 PASSED: one passive terrain routes both inlet speeds to their cups")


if __name__ == "__main__":
    main()
