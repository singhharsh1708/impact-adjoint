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
  on the input schema. That rule is necessary, not sufficient: what has to be
  resolved is the width of the window where the guard is negative, and near
  tangency that width scales as the square root of the penetration depth
  regardless of how wide the bump is. A grazing contact of depth 1e-6 is
  missed at step sizes more than an order of magnitude below what the rule
  suggests, and the run reports `status = 0` with one fewer impact rather than
  signalling anything. Optimizing bump amplitudes can steer a design toward
  tangency, so this is reachable rather than theoretical.
- Grazing impacts carry an inherent `δ^(-1/2)` sensitivity growth near
  tangency. The solver has a guard on the saltation denominator, but since
  that denominator scales as `δ^(1/2)` the guard is effectively unreachable:
  in practice you get a large finite gradient, and closer to tangency a
  silently missed event, rather than an error.
- The Jacobian is discontinuous across the `status` boundary as well as across
  bounce-count changes. A run that ends at `t_final` differentiates the state
  at a fixed time; a run truncated at the event budget differentiates it at a
  parameter-dependent event time. Straddling that boundary, `qf` moves
  continuously while its Jacobian can flip sign. The two cases are
  distinguishable only through `status`, which a client has to check.
- The tangential law removes a fixed fraction `μ` of tangential velocity at
  every impact, independent of the normal impulse. That is not merely "less
  than a full Coulomb cone": the implied friction coefficient grows without
  bound as the normal velocity goes to zero, so near-grazing and settling
  contacts are the regime where it is least physical. The Routh reset in
  future work is what fixes this.
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

## Known rough edges in the implementation

These were found by audit rather than by a failing test, and are recorded
because none of them is currently guarded:

- A degenerate crossing raises a Julia exception rather than returning a
  status code, so an optimizer line-search that overshoots into a graze kills
  the evaluation instead of getting a flagged result. `v_stop = 0` is a legal
  input that makes the settling branch unreachable and steers chatter toward
  that error.
- The trajectory history is accumulated on every step even when
  `n_samples = 0`, which is what every gradient call uses, and the step count
  is unbounded. A request with a very small `dt` and a long `t_final` will
  consume memory proportional to the step count.
- `t_end == t_final` holds to floating-point accumulation, not exactly, so a
  client comparing them should use a tolerance. The solver's own status field
  is the reliable signal.
- Four Jacobian evaluations per RK4 step recompute a matrix that is constant
  for this flow. Correct, but it is the dominant cost of every sensitivity
  call and the reason the VJP-to-apply ratio is as high as it is.

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
