"""Reproduce the chained-restart gradient reported in Diffrax issue #729.

The comparison table on the documentation site quotes three solver-dependent
gradients for the restart-after-event pattern. Every other number this
repository publishes is generated from a committed artifact, and that claim is
about somebody else's library, so it is held to the same standard rather than
being retyped from an issue thread.

This is the reproducer from
https://github.com/patrick-kidger/diffrax/issues/729, unchanged apart from
sweeping the solver and the step-size controller: an ODE whose right-hand side
switches at `event_time`,
solved up to a located event and then restarted from that state, differentiated
with respect to `event_time`. The state grows at rate 1 before the switch and
is constant after, so d(final state)/d(event_time) is 1.0.

Writes scripts/diffrax_event_gradient.json with the measured values and the
versions they were measured under.
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
SOLVERS = {
    "Heun": diffrax.Heun,
    "Tsit5": diffrax.Tsit5,
    "Bosh3": diffrax.Bosh3,
}


def final_state(event_time, solver_cls, clip):
    controller = diffrax.PIDController(rtol=1e-6, atol=1e-6)
    if clip:
        controller = diffrax.ClipStepSizeController(
            controller, jump_ts=[JUMP_TIME]
        )
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
    for clip in (True, False):
        label = "with ClipStepSizeController" if clip else "without it"
        print(f"  {label}:")
        out["gradients"][label] = {}
        for name, cls in SOLVERS.items():
            g = float(jax.grad(final_state)(JUMP_TIME, cls, clip))
            out["gradients"][label][name] = g
            ok = "matches" if abs(g - EXPECTED) < 1e-4 else "DIFFERS"
            print(f"    {name:6s} {g:+.7f}   {ok}")

    path = Path(__file__).parent / "diffrax_event_gradient.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.name} (diffrax {out['versions']['diffrax']}, "
          f"jax {out['versions']['jax']})")

    flat = [g for d in out["gradients"].values() for g in d.values()]
    assert any(abs(g - EXPECTED) > 1e-4 for g in flat), (
        "every solver now matches the expected gradient: issue #729 may be "
        "fixed, and the comparison table must be updated"
    )


if __name__ == "__main__":
    main()
