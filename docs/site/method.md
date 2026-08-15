# The method

## Architecture

```{mermaid}
flowchart LR
    P["design / material<br/>parameters θ"] --> A
    subgraph A["contact-sim · Julia Tesseract"]
        direction TB
        S["RK4 + event bisection"] --> V["forward variational X<br/>+ saltation jumps at impacts"]
    end
    A -- "qf, impact_x + VJP/JVP/Jacobian" --> B["score-target · JAX Tesseract<br/>(landing objective)"]
    B -- "loss" --> O["optax Adam / NumPyro NUTS"]
    O -- "one jax.grad via tesseract-jax" --> P
```

Two components that disagree about how to differentiate. One side uses
dual-number forward AD plus analytic event handling inside a Julia process,
the other reverse-mode tracing in JAX, and `tesseract-jax` composes them into
a single `jax.grad` anyway. The same solver container also serves NumPyro's
HMC over HTTP and a raw `curl` client, unchanged.

## The solver

`contact-sim` integrates a 2D point mass under gravity with optional linear
drag, over smooth terrain built from configurable Gaussian bumps. The guard is
`g(q) = y − h(x)`; at an impact the reset applies normal restitution `e` and
tangential retention `1 − μ` in the local terrain frame. Integration is RK4
with bisection-based event localization, plus interior guard probes so terrain
features narrower than one step are not stepped over.

## The sensitivities

Sensitivities propagate by the forward variational equation
$\dot{X} = (\partial f/\partial q) X$ on smooth segments. At each event, `X`
jumps:

$$
X^+ = R_q X^- + R_\theta - (f^+ - R_q f^-)\, \tau_\theta^\mathsf{T},
\qquad
\tau_\theta = -\frac{X^{-\mathsf{T}} g_q + g_\theta}{g_q \cdot f^-}
$$

This is the saltation jump condition applied to the θ-augmented system
($\dot\theta = 0$), which is the form Hiskens and Pai give (eqs. 57 to 59 with
the parameter augmentation of 62 to 63).

:::{note}
One detail worth flagging, since the most-cited statements of the saltation
matrix are written for time-varying guards: because parameters are constant
along the flow, $\partial g/\partial\theta$ enters the event-time **numerator**,
not the denominator where an explicit $\partial g/\partial t$ would sit.
:::

Local partials come from ForwardDiff dual numbers; the event structure is
handled analytically. From the assembled Jacobian the Tesseract serves
`jacobian`, `jacobian_vector_product` and `vector_jacobian_product`, so
`jax.grad`, `jax.jvp` and `jax.jacrev` all work through it.

## Termination semantics

When a trajectory leaves the model's validity region, either the event
capacity or Zeno chatter, the solver terminates at the event with an explicit
status and returns the **total derivative at that point**, event-time
dependence included. Optimizers therefore never consume silently nonphysical
states. The alternative, integrating on and ignoring events, produces
plausible-looking gradients of a ball in free-fall below the terrain.

| status | meaning | what `qf` is |
|---|---|---|
| 0 | ran to `t_final` | state at `t_final` |
| 1 | event capacity reached | pre-impact state at the truncating crossing |
| 2 | contact settled below `v_stop` | post-impact state at that impact |

## Cost

The sensitivities are forward-variational, so a VJP costs O(n_params):
measured at 6.0x a forward solve for 14 parameters and 8.5x for 77, and affine
in parameter count out to 581 (93 microseconds per parameter, R² = 0.998). See
[verification studies](studies.md). At thousands of design variables the
natural extension is a reverse-mode saltation adjoint behind the same
endpoint.
