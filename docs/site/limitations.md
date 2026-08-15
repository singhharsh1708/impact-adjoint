# Limitations and future work

## Scope of the model

- The impact law is kinematic Newton restitution plus a tangential retention
  factor, not a full Coulomb friction cone. Sticking and sliding contact modes
  are out of scope, and trajectories entering them terminate with `status=2`
  rather than being silently integrated.
- Gradients are exact for the continuous hybrid system at fixed event
  topology. Across bounce-count boundaries the objective is discontinuous,
  which is inherent to contact and visible in E1.
- Event detection resolves terrain features wider than `|vx|·dt/3`, documented
  on the input schema. Grazing impacts carry an inherent `δ^(-1/2)`
  sensitivity growth near tangency.
- Two dimensions, single body. The saltation machinery is dimension-agnostic;
  the scope is a deliberate trade for verified correctness inside the
  hackathon period.

## Honest caveats on the results

- The separator's low-restitution particle is stopped by the event budget
  rather than by coming to rest, so its separation surface is "position after
  eight impacts", not "position at rest". A fixed-length chute imposes an
  analogous cut, though not the identical one.
- The optimizer comparison reverses under wall-clock accounting. Both
  accountings are reported in [studies](studies.md).
- Designing environment geometry through contact-driven simulation is not new
  in itself. What we could not find in the literature is design using exact
  event-time sensitivities and targeting per-impact routing rather than a bulk
  flow statistic.

## Future work

**Reverse-mode saltation adjoint.** The sensitivities here are forward
variational, so VJP cost scales with parameter count, measured at 93
microseconds per parameter. A backward costate integration with `Sᵀ` jumps at
events would serve thousands of design variables behind the same endpoint,
with no change for the client, and would close the wall-clock gap in the
benchmark.

**Coulomb friction cone.** The impulse-ratio (Routh) reset is a small change
to the reset map, at the cost of re-deriving the closed-form oracle, and would
extend the model into sticking and sliding contact.

**Three dimensions and multiple bodies.** The work is in guard geometry and
event bookkeeping, not in the sensitivities.

**Upstream.** Test the tesseract-jax fix for
[issue 234](https://github.com/pasteurlabs/tesseract-jax/issues/234) against
this solver once it lands.
