# The problem: gradients die at events

Differentiable simulation has a failure mode worse than an exception. It
returns confident, finite, wrong gradients, and it does so wherever the
dynamics are event-driven: contact, impact, switching, thresholds.

The reason is structural. The parameter-dependence of the *event time*
contributes a term to the sensitivity, the saltation term, that autodiff of a
time-stepping program cannot see, because the step at which the event fires is
an integer and is piecewise-constant in the parameters.

## Measured, three ways

```{image} ../figures/e3_bias.png
:alt: what autodiff returns at an impact
:width: 100%
```

A pure-JAX simulation of a bouncing ball (RK4 scan, reset applied at the grid
point via `jnp.where`) converges to the correct *trajectory* as `dt` shrinks,
and reports `d x(T)/d v0y = 0.0` at every resolution. The true value is
`+0.0904`, pinned independently by the symbolic closed form used in the
[verification studies](studies.md).

The exact zero is specific to this configuration, state-independent resets over
flat terrain. On curved terrain the same program returns a nonzero wrong value
instead, which is harder to notice.

## The steelman

The honest repair inside pure JAX is to interpolate the crossing time from the
guard and reset at the interpolated state. That recovers a converging gradient,
and it is the point rather than a counterexample: the repair *is* first-order
event-time sensitivity machinery, hand-implemented. Its error is dt-dependent
and non-monotone (2×10⁻³ down to 6×10⁻⁵ across our sweep), and every new
guard and reset pair owes its own derivation.

## What JAX already does

To be precise about the ecosystem, since this is the first question a JAX user
asks. [Diffrax](https://docs.kidger.site/diffrax/) has differentiated event
times since v0.6.0, by implicit differentiation through an Optimistix root
find, so a single event is handled natively and correctly.

What is missing is the rest of a hybrid trajectory. `diffrax.Event` terminates
a solve and has no reset map, so a multi-impact chain has to be assembled by
restarting the solver after each event and applying the reset in user code.
That route is expressible, and it is currently unreliable:
[Diffrax issue 729](https://github.com/patrick-kidger/diffrax/issues/729)
reports exactly this pattern returning solver-dependent wrong gradients (0.50
with Heun, -1.42 with Tsit5, 0.78 with Bosh3, against a true value of 1.0),
and the maintainer's fix branch is unmerged.

Rather than build on that, impact-adjoint puts the event-aware machinery
behind a component boundary. See [the method](method.md).
