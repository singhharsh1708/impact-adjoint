# How this relates to other differentiable simulators

The question a reader should ask first is whether this problem is already
solved elsewhere. Partly it is. This page states what each neighbouring system
does at a contact and what its gradient through an impact actually is, citing
each project's own documentation or paper rather than characterising it from
the outside.

Two things are worth saying before the table. The failure this project starts
from is not new: DiffTaichi described it in 2019, in the same terms, and fixed
it for its own simulators. And a 2022 evaluation of contact gradients across
seven differentiable engines concluded plainly that "the gradients are not
always correct" ([Zhong, Han & Brikis, arXiv:2207.05060](https://arxiv.org/abs/2207.05060)),
which is the honest backdrop for every row below.

```{list-table}
:header-rows: 1
:widths: 16 34 50

* - System
  - What it does at a contact
  - What the gradient through an impact is
* - [Diffrax](https://docs.kidger.site/diffrax/)
  - No contact model. A general ODE/SDE solver whose events "allow for
    interrupting a differential equation solve, by terminating the solve
    before `t1` is reached"
    ([events](https://docs.kidger.site/diffrax/api/events/)).
  - Event *times* are differentiable: v0.6.0 added continuous events, locating
    the crossing with a root find and handling the event time's parameter
    dependence through the implicit function theorem. There is no reset map,
    so a bounce has to be assembled by restarting the solve and applying the
    reset in user code, and that pattern is currently unreliable:
    [issue #729](https://github.com/patrick-kidger/diffrax/issues/729). We
    reproduce it below rather than quoting it. The maintainer's fix branch,
    `partway-event-interpolate-to-step`, is unmerged.
* - [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
  - MuJoCo's soft constraint solver, contact resolved as a convex program.
  - The documentation states that differentiability "is mostly supported in
    MJX-JAX but is not currently available in MJX-Warp", and that "MJX-Warp
    does not support automatic differentiation and has no immediate plans to
    support auto-diff."
* - [Brax](https://github.com/google/brax)
  - Several backends, including positional, spring and generalized dynamics.
  - Differentiable end to end. Brax is one of the engines evaluated in
    arXiv:2207.05060, whose finding is that gradients through contact are not
    always correct across the formulations it surveys.
* - [DiffTaichi](https://arxiv.org/abs/1910.00935)
  - Source-to-source AD over a simulation DSL, with continuous collision
    detection.
  - The paper documents exactly this failure and repairs it: differentiating
    the discretized program naively means "no matter how small Δt is, the
    evaluated gradient of final height w.r.t. initial height will be 1 instead
    of −1", because "time discretization itself is not differentiated by the
    compiler". Adding a precise time of impact corrects it. This is the
    closest prior art to the mechanism used here.
* - [NVIDIA Warp](https://github.com/NVIDIA/warp)
  - A differentiable kernel language rather than a contact model. `warp.sim`
    was removed in
    [v1.10.0](https://github.com/NVIDIA/warp/releases/tag/v1.10.0), superseded
    by [Newton](https://developer.nvidia.com/newton-physics).
  - Gradients propagate through Warp kernels. Contact behaviour, and therefore
    the gradient at an impact, now belongs to Newton rather than to Warp
    itself; Warp is one of the engines covered by arXiv:2207.05060.
* - [Dojo](https://arxiv.org/abs/2203.00806)
  - Hard contact and friction as a nonlinear complementarity problem with
    second-order cone constraints, solved by a custom primal-dual
    interior-point method.
  - Smooth gradients obtained through the implicit function theorem, with the
    interior-point central-path parameter exposed as a user-tunable knob: a
    high value gives a smooth approximation with smooth gradients, a low value
    gives hard contact and precise rollouts.
* - **impact-adjoint**
  - Newton restitution with a tangential retention factor, applied at a
    located crossing rather than at a grid point.
  - The classical saltation matrix, applied analytically at each impact, so
    the event time's parameter dependence enters the sensitivity exactly and
    at any step size. Served across a Tesseract boundary, so the calling
    program does no event handling of its own. Exact at fixed event topology;
    across a bounce-count change the objective is genuinely discontinuous, and
    that is stated in [limitations](limitations.md).
```

## The Diffrax restart, measured

The row above is the one quantitative claim on this site about somebody else's
library, so it is generated from a committed artifact like everything else
rather than retyped from an issue thread.

The reproducer is the one from issue #729: an ODE whose right-hand side
switches at `event_time`, solved up to a located event and restarted from that
state, differentiated with respect to `event_time`. The state grows at rate 1
before the switch and is flat after it, so the derivative is exactly `1.0`.

```{diffrax-table}
```

Two things are worth being careful about here. With the clipping controller
every solver returns exactly zero, which is the same failure this project
starts from, reached by a different route. Without it the answer is wrong in a
solver-dependent way instead.

Neither set matches the numbers quoted in the issue thread itself
(`0.5`, `-1.4211714`, `0.7777778`), which were reported under different
versions; that is why the versions are printed above and why we publish what we
measured rather than what we read. This has not been raised with the Diffrax
maintainer, so treat it as a reproduction of a known open issue and not as a
new finding.

## What is actually different here

Not the observation. DiffTaichi made it in 2019 and Dojo, Nimble and others
have all engaged with gradients at contact since.

Three things are different in combination, and the honest claim is the
combination rather than any one of them:

1. **The event-time term is analytic, not relaxed and not interpolated.** Dojo
   obtains smooth gradients by relaxing hard contact through the central-path
   parameter; DiffTaichi's time of impact is a correction inside the
   simulator's own AD. Here the jump condition is the saltation matrix applied
   in closed form, so the gradient is exact at any `dt` rather than converging
   as `dt` shrinks.
2. **It lives behind a component boundary.** The sensitivity is computed in
   Julia and consumed by a JAX program that contains no event logic, no reset
   map and no knowledge that impacts occur. Neither system has to adopt the
   other's autodiff.
3. **It is checked against solver-independent oracles.** A symbolic closed
   form and an independent scipy reimplementation, both in
   [studies](studies.md), rather than self-consistency alone.

The scope is correspondingly narrow: two dimensions, a single body, no
friction cone. Every system in the table above does more physics than this one
does.
