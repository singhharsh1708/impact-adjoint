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
  - Event *times* are differentiable and correct. v0.6.0 added continuous
    events, locating the crossing with a root find and handling the event
    time's parameter dependence through the implicit function theorem. We
    measured the restart-after-event pattern ourselves and Diffrax returns the
    exact gradient under all three solvers tried; see below. What it does not
    provide is a *reset map*, so a bounce must be assembled by restarting the
    solve and applying the impact in user code, and the jump time must be
    closed over differentiably by hand. That is an ergonomic difference, not a
    correctness one.
* - [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html)
  - MuJoCo's soft constraint solver, contact resolved as a convex program.
  - The documentation states that differentiability "is mostly supported in
    MJX-JAX but is not currently available in MJX-Warp", and that "MJX-Warp
    does not support automatic differentiation and has no immediate plans to
    support auto-diff."
* - [Brax](https://github.com/google/brax)
  - Several backends, including positional, spring and generalized dynamics.
  - Differentiable end to end, though its own README now states that only
    `brax/training` is actively maintained as of 0.13.0 and points users at
    MJX for physics. Brax is one of the engines evaluated in
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

:::{important}
**This section previously said the opposite, and was wrong.** An earlier
version of this page claimed the restart-after-event pattern returned
solver-dependent wrong gradients, citing three numbers from
[issue #729](https://github.com/patrick-kidger/diffrax/issues/729). Measuring
it properly shows Diffrax is correct and the fault was in how the reproducer
was written, including ours. The correction is kept here rather than quietly
removed.
:::

The reproducer is the one from issue #729: an ODE whose right-hand side
switches at `event_time`, solved up to a located event and restarted from that
state, differentiated with respect to `event_time`. The state grows at rate 1
before the switch and is flat after, so the derivative is exactly `1.0`.

```{diffrax-table}
```

The first row is the documented usage and it is exact. The other two are
mistakes a caller can make: passing the jump time to
`ClipStepSizeController` as a plain Python float, so the controller never
closes over it differentiably, or not declaring the jump at all, which feeds
the solver a discontinuous vector field it does not accept as valid input.

Both are what the maintainer identified in the thread, telling the reporter to
"replace ... `jump_ts=[jump_time]` with ... `jump_ts=[event_time]`, to close
over the jump time differentiably", and separately that "your vector field is
discontinuous, so this already isn't valid input for Diffrax". The reporter
subsequently wrote that the minimal example may not reproduce the problem they
were chasing. The three numbers we had quoted are one column of a two-column
table in a single comment, measured on an experimental branch; the omitted
column has Heun returning exactly `1.0`.

The honest summary is that Diffrax differentiates event times correctly, and
the difference here is that it has no reset map, so the bounce and the
differentiable jump time are the caller's job rather than the component's.

## What is actually different here

A prior-art sweep of this question was unkind to the first version of this
section, which claimed three things. Two of them do not survive, and they are
withdrawn rather than softened.

**Not the analytic event-time sensitivity.** Saltation matrices for hybrid
sensitivity are classical, and the word is in our own description of the
method. They have been used for optimization and for design specifically:
[Kong, Payne, Zhu and Johnson (Proc. IEEE 2024)](https://arxiv.org/abs/2306.06862)
states that adapted saltation conditions "were used to formulate and solve
optimal design problems", and there is a pip-installable package,
[hybrid-tools](https://github.com/robomechanics/hybrid-tools), providing
saltation matrices with adjoint-gradient trajectory optimization and a
bouncing-ball example. Exact parametric sensitivity across state-dependent
events goes back to Galán, Feehery and Barton (1999) and Barton and Lee
(2002).

**Not exact event-time sensitivities in existing software.**
[SciMLSensitivity.jl](https://github.com/SciML/SciMLSensitivity.jl) implements
the event-time correction for continuous callbacks in its adjoint, and its
documented example is a bouncing ball whose restitution coefficient is
optimized by gradient descent. Diffrax gained autodifferentiable event
handling with pathwise event-time gradients through
[Holberg and Salvi (NeurIPS 2024)](https://arxiv.org/abs/2405.13587).

**Not derivatives across a component boundary.** FMI has shipped directional
derivatives since 2.0 (2014) and adjoint derivatives since 3.0 (2022), the
latter motivated explicitly by machine-learning frameworks wanting VJPs across
a binary boundary. Tesseract's own `fortran_enzyme` example is this pattern
with a different language. This hackathon's premise is composing
differentiable components across languages, so treating it as a contribution
would be claiming credit for meeting the entry criteria. It is plumbing, and
it is the right plumbing, but it is not new.

**What we could not find occupied** is the conjunction, and it is worth
stating with its parts conceded. Each ingredient is established: event-time
sensitivities since 1958; gradient routing of a single trajectory through
impacts, as in DiffTaichi's `billiards` example, which backpropagates through
a chain of collisions with a time-of-impact correction to steer one ball to
one target; sampling-based design of passive geometry for per-object impact
routing, in [Roussel et al. (SIGGRAPH 2019)](https://dl.acm.org/doi/10.1145/3306346.3322983)
and in Berkowitz and Canny (ICRA 1996), whose objective is per-object and
whose method is grid enumeration; and restitution-based separators as
physical devices, including US 8,640,879 for an inclined chute sorting rubber
from plastic by rebound.

What appears unoccupied is their combination: gradient-based design of the
*passive geometry itself*, using exact rather than relaxed event times, with a
*per-object routing* objective rather than a bulk flow statistic. The nearest
neighbours make the gap narrow. Changing DiffTaichi's billiards decision
variable from the initial velocity to the peg positions is a small edit;
Roussel et al. design geometry for ordered impacts but by sampling rather than
gradients. We claim the conjunction, not any element of it.

The scope is correspondingly narrow: two dimensions, a single body, no
friction cone. Every system in the table above does more physics than this one
does.
