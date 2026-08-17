"""What Diffrax actually does with a gradient through a restart-after-event.

An earlier version of this script, and the comparison page that quoted it,
claimed the restart-and-reset pattern returned solver-dependent wrong
gradients, citing three numbers from Diffrax issue #729. That was wrong, and
this script now measures why.

Reading the whole thread rather than the one comment: the maintainer diagnosed
the reproducer as a usage error. `ClipStepSizeController(..., jump_ts=[...])`
was given a plain Python float for the jump time, so the controller did not
close over it differentiably:

    "You should replace ... jump_ts=[jump_time] with ... jump_ts=[event_time],
     to close over the jump time differentiably. Job done!"

and separately, that a discontinuous vector field is not valid input unless
the jump is declared through the controller. The reporter later wrote that the
minimal example may not reproduce the problem they were chasing. The three
numbers we quoted (0.5, -1.4211714, 0.7777778) are one column of a two-column
table measured on an experimental branch, and the omitted column has Heun
returning exactly 1.0.

So this sweeps the usage rather than the solver. The ODE grows at rate 1 until
`event_time` and is flat after, so d(final state)/d(event_time) is exactly 1.0.

Writes scripts/diffrax_event_gradient.json.
"""

import json
import platform
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import diffrax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optimistix as optx  # noqa: E402

JUMP_TIME = 0.98
EXPECTED = 1.0
SOLVERS = {"Heun": diffrax.Heun, "Tsit5": diffrax.Tsit5, "Bosh3": diffrax.Bosh3}

USAGES = {
    "jump time closed over differentiably": "traced",
    "jump time passed as a constant": "float",
    "no jump declared to the controller": "none",
}


def final_state(event_time, solver_cls, usage):
    controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    if usage == "traced":
        controller = diffrax.ClipStepSizeController(
            controller, jump_ts=jnp.asarray([event_time])
        )
    elif usage == "float":
        controller = diffrax.ClipStepSizeController(controller, jump_ts=[JUMP_TIME])

    term = diffrax.ODETerm(
        lambda t, y, args: jnp.array(
            [jnp.select([jnp.less(t, event_time), True], [1.0, 0.0])]
        )
    )
    solver = solver_cls()
    sol_event = diffrax.diffeqsolve(
        term, solver, t0=0, t1=2, dt0=None, y0=jnp.array([0.0]),
        stepsize_controller=controller,
        event=diffrax.Event(
            cond_fn=lambda t, y, args, **kw: event_time - t,
            root_finder=optx.Newton(atol=1e-4, rtol=1e-4),
        ),
        max_steps=100,
    )
    sol = diffrax.diffeqsolve(
        term, solver, t0=sol_event.ts[-1], t1=2, dt0=None, y0=sol_event.ys[-1],
        stepsize_controller=controller, max_steps=100,
    )
    return sol.ys[-1, 0]


def main():
    out = {
        "expected": EXPECTED,
        "jump_time": JUMP_TIME,
        "versions": {
            "diffrax": diffrax.__version__,
            "jax": jax.__version__,
            "optimistix": optx.__version__,
            "python": platform.python_version(),
        },
        "gradients": {},
    }
    print(f"d(final state)/d(event_time), expected {EXPECTED}")
    for label, usage in USAGES.items():
        out["gradients"][label] = {}
        print(f"  {label}:")
        for name, cls in SOLVERS.items():
            try:
                g = float(jax.grad(final_state)(JUMP_TIME, cls, usage))
            except Exception as exc:
                g = None
                print(f"    {name:6s} raised {type(exc).__name__}")
            if g is not None:
                out["gradients"][label][name] = g
                ok = "correct" if abs(g - EXPECTED) < 1e-4 else "WRONG"
                print(f"    {name:6s} {g:+.7f}   {ok}")

    path = Path(__file__).parent / "diffrax_event_gradient.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.name} (diffrax {out['versions']['diffrax']}, "
          f"jax {out['versions']['jax']})")

    correct = out["gradients"]["jump time closed over differentiably"]
    assert all(abs(g - EXPECTED) < 1e-4 for g in correct.values()), (
        "Diffrax no longer returns the correct gradient under the documented "
        "usage; the comparison page says it does and must be revisited"
    )


if __name__ == "__main__":
    main()
