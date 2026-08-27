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
  tangency. The solver raises if the saltation denominator falls below
  `1e-12`, but that denominator scales as `δ^(1/2)` and `δ` cannot fall below
  the guard's own floating-point resolution, so on metre-scale terrain it
  floors near `1e-7`, five orders above the threshold. Tangency therefore
  never raises. What you get instead is a large finite gradient, measured
  growing from 54 to 8e6 as `δ` falls from `1e-2` to `1e-13`; then, for grazes
  shallow enough that the rebound falls under `v_stop`, a `status = 2` report;
  and below that a silently missed event at `status = 0`.
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
- The wall-clock ordering reversed when we made our own gradient cheaper: Adam
  is ahead on all five seeds under both accountings now. That is evidence about
  this implementation, not about gradient methods against gradient-free ones.
  Both accountings are reported in [studies](studies.md).
- The gradient is not demonstrated to be the better search method on this
  problem. At a matched budget on the E5b ensemble objective, tuned CMA-ES
  reaches a better design than Adam, and on the E5 held-out purity Nelder-Mead
  matches Adam with a wider margin. What buys the robustness is optimising the
  ensemble objective, not the gradient that optimises it. See
  [studies](studies.md).
- The separator is defined at its own horizon: evaluated at `t_final = 3.0`
  instead of 2.2, the trained `e = 0.8` particle lands 0.67 m from its target
  rather than 0.42 mm.
- Designing environment geometry through contact-driven simulation is not new
  in itself. What we could not find in the literature is design using exact
  event-time sensitivities and targeting per-impact routing rather than a bulk
  flow statistic.

## Known rough edges in the implementation

These were found by audit rather than by a failing test, and are recorded
because none of them is currently guarded:

- A degenerate crossing raises a Julia exception rather than returning a
  status code. Tangency does not reach it, for the reason given above, and
  neither does chatter: the `1e-13` lift applied after each reset so detection
  re-arms also floors the approach speed at `1.4e-6`, which is where a
  settling sequence's denominator stops falling. What does reach it is a
  launch height small enough that the first impact speed `sqrt(2 g y0)` falls
  under `1e-12`. `y0 = 1e-26` is accepted by the schema and raises instead of
  returning a status, and `y0` is checked only for finiteness. `v_stop = 0` is
  separately a legal input that makes the settling branch unreachable, so a
  run that would have settled reports `status = 1` at the event budget.
- A second guard rejects a crossing whose velocity is not approaching. It
  fires only when the fixed RK4 step amplifies the drag mode instead of
  damping it, so the located crossing is an artifact of the integrator rather
  than of the trajectory. Bisecting the threshold gives `drag * dt = 2.7898`
  at `dt = 1e-3` and `2.7859` at `dt = 1e-4`, converging on the RK4 stability
  limit `2.785293563`. The schema rejects `drag * dt` past that limit, so this
  one is a backstop for direct Julia use rather than a reachable client
  failure.
- The trajectory history is accumulated on every step even when
  `n_samples = 0`, which is what every gradient call uses, and the step count
  is unbounded. A request with a very small `dt` and a long `t_final` will
  consume memory proportional to the step count.
- `t_end == t_final` holds to floating-point accumulation, not exactly, so a
  client comparing them should use a tolerance. The solver's own status field
  is the reliable signal.

## Future work

**Reverse-mode saltation adjoint.** This used to head the list, on the grounds
that forward-variational VJP cost scales with parameter count and was measured
at 93 microseconds per parameter. That figure turned out to be an artifact of
the implementation rather than of the method: the flow is affine in the state,
so the RK4 variational update collapses to a fixed tangent map that can be
composed across a smooth segment and applied only at events. With that
factoring the slope is 15 microseconds per parameter
and a VJP at 581 parameters costs 1.8x a forward solve,
so the parameter-count argument for an adjoint is largely gone.

It remains the right extension for large *state* dimension. The tangent map is
4x4 here; in three dimensions with multiple bodies it is not, and a backward
costate integration with `Sᵀ` jumps at events would then be the cheaper side of
the trade. That is the justification to keep, and it is a different one from
the one we started with.

**Coulomb friction cone.** The impulse-ratio (Routh) reset is a small change
to the reset map, at the cost of re-deriving the closed-form oracle, and would
extend the model into sticking and sliding contact.

**Three dimensions and multiple bodies.** The work is in guard geometry and
event bookkeeping, not in the sensitivities.

**Upstream.** Test the tesseract-jax fix for
[issue 234](https://github.com/pasteurlabs/tesseract-jax/issues/234) against
this solver once it lands.
