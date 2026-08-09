# impact-adjoint: end-to-end gradients through impact events

**Track 1 — Inverse design & shape optimization** (cross-listed: Track 4,
differentiable inference). Solo entry.

**In one sentence:** we differentiate *through impacts* — exactly, not by
smoothing — and use it to design passive structures whose function only exists
because of those impacts: a bounce separator that sorts particles by material,
a terrain that routes different inlet speeds to different bins, and Bayesian
recovery of material parameters from where things actually hit. Impact-driven
sorting is real machinery (bounce/impact separators sort PET from rubber by
resilience); simulating it differentiably is exactly where standard autodiff
fails silently.

## 1. The problem: gradients die at events, quietly

Differentiable simulation has a well-known failure mode that is worse than an
exception: it returns confident, finite, *wrong* gradients. It happens wherever
the dynamics are event-driven — contact, impact, switching, thresholds. The
parameter-dependence of the *event time* contributes a term to the sensitivity
(the saltation term) that the autodiff of a time-stepping program structurally
cannot see: the step at which the event fires is an integer, piecewise-constant
in the parameters.

Experiment E3 makes this concrete. A pure-JAX simulation of a bouncing ball
(RK4 scan, reset applied at the grid point via `jnp.where`) converges to the
correct *trajectory* as `dt → 0` — and reports `d x(T)/d v0y = 0.0` at every
resolution. The true value is `+0.0904`, confirmed by two independent
witnesses. The bias is O(1) and refinement-proof:

![E3](figures/e3_bias.png)

This is not a strawman implementation; it is what `jax.grad` of the natural
program does. Fixing it requires event-time sensitivity machinery — a different
*differentiation semantics*, not a smaller `dt`.

## 2. What we built

A two-Tesseract pipeline in which the event-aware machinery lives behind a
Tesseract boundary, and JAX never needs to know:

```
(v0, e, mu, terrain θ)                                            optax Adam
      │                                                                ▲
      ▼                                                                │
┌────────────────────┐   qf, impact_x   ┌───────────────────┐   loss   │
│ contact-sim        │ ───────────────▶ │ score-target      │ ─────────┘
│ Julia · RK4+events │                  │ JAX · autodiff    │
│ variational + salt-│                  │ (cup objective)   │
│ ation sensitivities│                  └───────────────────┘
└────────────────────┘
        one jax.grad, via tesseract-jax
```

**contact-sim** (Julia): a 2D point mass under gravity (optional linear drag)
over smooth terrain `h(x)` (three Gaussian bumps). Guard `g(q) = y − h(x)`;
impact reset applies normal restitution `e` and tangential retention `1 − μ`
in the local terrain frame. Integration is RK4 with bisection-based event
localization (plus interior guard probes for sub-step terrain features).

Sensitivities propagate by the forward variational equation `Ẋ = (∂f/∂q) X`
on smooth segments; at each event, `X` jumps:

```
X⁺ = R_q X⁻ + R_θ − (f⁺ − R_q f⁻) τ_θᵀ,     τ_θ = −(X⁻ᵀ g_q + g_θ) / (g_q · f⁻)
```

— the saltation update. Local partials (`∂f/∂q`, `g_q`, `g_θ`, `R_q`, `R_θ`)
come from ForwardDiff dual numbers; the event structure is handled
analytically. From the assembled Jacobian, the Tesseract serves `jacobian`,
`jacobian_vector_product`, and `vector_jacobian_product`, so `jax.grad`,
`jax.jvp`, and `jax.jacrev` all work through it.

**score-target** (JAX): a differentiable landing objective (quadratic distance
to a cup plus kinetic penalty), built with the `tesseract init --recipe jax`
endpoints.

### Why this needs Tesseract

The two components disagree about how to compute *and* how to differentiate:
dual-number forward AD plus analytic event handling in a Julia process, versus
reverse-mode tracing autodiff in JAX. There is no way to express the saltation
update inside `jax.grad`'s programming model without reimplementing the solver
as a JAX custom primitive with hand-written VJPs — at which point you have
written a worse, unshareable Tesseract. The Tesseract contract (typed schemas +
gradient endpoints) is exactly the seam: Julia publishes *what its derivatives
are*, not *how it got them*, and `tesseract-jax` splices them into JAX's chain
rule. The solver container is reusable as-is from any other client (PyTorch via
tesseract-torch, probabilistic stacks for Track 4, CLI).

A deliberate design point: when a trajectory leaves the model's validity region
— more impacts than the schema's capacity, or chatter toward the Zeno
accumulation — the solver *terminates at the event with an explicit status*
and returns the **total derivative at the truncation point** (event-time
dependence included). Optimizers therefore never consume silently nonphysical
states, and gradient descent remains well-posed even at the model's edges.
Robustness here is a gradient-correctness feature: our adversarial testing
showed that the naive alternative (integrate on, ignore events) produces
plausible-looking gradients of a ball in free-fall below the terrain.

## 3. Gradients doing real work

**E1 — inverse design.** Optimize launch velocity and restitution so the ball
lands in a cup at `x = 4.3` on bumpy terrain, 1.1 m past where the initial
guess settles. Adam, lr 0.03, 150 iterations, every step one `jax.grad`
through both Tesseracts:

![E1](figures/e1_trajectory.png)

Miss distance falls **1.12 m → 2.7 cm**, through five impacts. Notably, the
optimizer crosses bounce-count boundaries (4 → 6 → 5 bounces) during descent —
the objective is discontinuous there (inherent to contact), and the
fixed-topology gradients on each side are exact, which is what carries Adam
through:

![E1 convergence](figures/e1_convergence.png)

**E2 — calibration (Track 4 flavor).** Recover material parameters `(e, μ)`
from the *positions of three impacts* observed with 5 mm noise. The observable
exists only because of events — there is no smooth surrogate for "where it
hit". Gradient descent through the solver's VJP recovers `e` to 0.002 and `μ`
to 0.009 (noise-limited) from a distant start `(0.5, 0.3)`.

**E2b — Bayesian calibration (Track 4 proper).** The same inverse problem as a
posterior: NumPyro's NUTS sampler, whose Hamiltonian dynamics require a
JAX-differentiable log-density, runs directly against the *containerized*
solver — every leapfrog step calls the Tesseract's apply and saltation-VJP
endpoints over HTTP. Posterior: `e = 0.697 ± 0.008`, `μ = 0.096 ± 0.010`;
the truth lies within one standard deviation of both marginals, and the
posterior resolves the physically meaningful e–μ ridge (more bounce traded
against more tangential loss):

![E2b](figures/e2b_posterior.png)

This is the composition Track 4 asks for: an expensive event-driven solver
dropped into a probabilistic workflow unchanged. (One engineering detail makes
it possible: NumPyro jits its NUTS step, and jitted JAX callbacks may execute
off the main thread — which an in-process embedded Julia runtime does not
tolerate. The *container* is what turns that from a threading bug into a
non-issue: HMC's callbacks are plain HTTP calls. The same solver was also
driven with nothing but `curl` — `scripts/second_client_curl.sh` — so
"reusable from any client" is demonstrated, not asserted.)

**E4 — terrain design (shape optimization proper).** The terrain parameters
are differentiable inputs, so the *structure* can be the design variable. A
single terrain is optimized so that balls entering at 1.6 m/s and 2.6 m/s are
routed to two different cups: miss distances fall from 2.88 m / 0.48 m to
**2.2 cm / 3.0 cm**. The optimizer flattens two bumps and keeps one narrow
deflector that the slow ball cannot clear — the sorting logic ends up encoded
in the geometry.

**E5 — bounce separator, 24-dimensional, head-to-head (the headline).**
Industrial bounce/impact separators sort particles by resilience. We design
one: two particles enter *identically* and differ only in restitution
(`e = 0.5` "rubber" vs `e = 0.8` "PET"); the design vector is the 24 bump
amplitudes of the surface; each particle must land in its own bin.

![E5](figures/e5_separator.png)

The designed surface separates the materials from a shared first impact —
rubber dies into bin A within 8 bounces, PET carries over the designed hills
into bin B (the optimizer grew a backstop that bounces PET *backward* into its
bin) — with landing errors of 0.4 mm and < 0.1 mm. The right panel is the
"gradients doing real work" claim made quantitative: under one evaluation
budget (a gradient call charged double), Adam on the saltation gradients
reaches an objective of **2×10⁻⁷**, while CMA-ES plateaus at 2×10⁻³ (9,200×
worse) and Nelder-Mead at 2×10⁻² — in 24 dimensions, exact gradients through
the events are not a convenience but the difference between solving the design
problem and not solving it.

## 4. Correctness: independent oracles, not self-agreement

FD-vs-analytic agreement through the same solver proves consistency, not
correctness. We therefore validated against two solver-independent oracles:

1. **Closed form** (flat terrain): the full multi-bounce trajectory and its
   Jacobian derived symbolically in rational arithmetic — including a
   hand-differentiated impact-position recursion and an implicit-function-
   theorem cross-check for terrain sensitivities. Solver agreement: primals
   ~1e-12, **Jacobian 7e-12**.
2. **Cross-implementation** (bumpy terrain, with and without drag): the spec
   reimplemented from scratch on scipy's adaptive RK45 with its own event
   localization. Primal agreement 1e-10; the solver's analytic Jacobian
   matches finite differences *through the independent implementation* to
   **5e-9** — covering exactly the sloped-contact-frame and drag sectors the
   flat oracle cannot reach.

The regression suite additionally pins: energy conservation at `e = 1`
(relative drift 5e-13 across five impacts — RK4 is exact on ballistic arcs and
events are localized to machine precision), chatter termination, event-capacity
truncation, sub-step terrain-feature detection, and `jax.grad` end-to-end
equality with FD for all seven differentiable inputs.

Near-grazing behavior matches theory: impact sensitivities grow as
`δ^(−1/2)` approaching tangency, and analytic-vs-FD agreement holds down to
`δ ≈ 1e-8`, beyond which FD itself is invalid.

## 5. Related work: how everyone else gets contact gradients

Existing differentiable engines obtain contact gradients in one of three ways:
**AD through smoothed or penalty-based stepping** (Brax [Freeman et al.,
NeurIPS 2021], DiffTaichi [Hu et al., ICLR 2020], gradSim [Jatavallabhula et
al., ICLR 2021]); **implicit differentiation of a relaxed complementarity
solve** (Dojo [Howell et al., arXiv 2022], Nimble [Werling et al., RSS 2021]);
or **learned contact models** (Zhong et al., NeurIPS 2021). Each trades
gradient fidelity at the event for a differentiation strategy its host
framework can express. The ML community has documented the cost: Suh et al.
(ICML 2022, "Do Differentiable Simulators Give Better Policy Gradients?") show
first-order gradients through contact can be biased or high-variance, and
Antonova et al. (CoRL 2022) show contact-rich loss landscapes defeat naive
local descent.

The exact alternative is classical: the jump condition for trajectory
sensitivities at a switching surface — the **saltation matrix** — dates to
Aizerman & Gantmacher (1958), was given its modern hybrid-systems treatment by
Hiskens & Pai (IEEE TCAS 2000), and is surveyed for robotics by Kong, Payne,
Zhu & Johnson (Proc. IEEE, 2024). Our contribution is not the formula; it is
the *packaging*: the saltation machinery lives in a Julia process and is
published through Tesseract's gradient endpoints, so any AD framework consumes
exact event sensitivities without being able to express them natively —
resolving precisely the mismatch the three engine families work around.

To our knowledge, no prior work optimizes the *environment/structure* through
contact events (nearest neighbors: contact-aware robot morphology design, Xu
et al., RSS 2021; differentiable soft-body design, Geilinger et al., TOG
2020). E4/E5's terrain-as-design-variable problems appear to be new.

## 6. Limitations

- The impact law is kinematic Newton restitution + tangential retention, not a
  Coulomb cone; sticking/sliding contact modes are out of scope (explicit
  `status=2` termination instead of pretending).
- Gradients are exact at fixed event topology; the objective is discontinuous
  across bounce-count boundaries (physical, and visible in E1).
- Event detection resolves terrain features wider than `|vx|·dt/3`; the
  constraint is documented on the input schema.
- 2D, single body. The saltation machinery is dimension-agnostic; the scope is
  a deliberate trade for verified correctness within the hackathon period.

## 6. Reproducibility

Everything (validation, experiments, figures) reproduces with the commands in
the README, CPU-only, in minutes. The Docker images build for arm64 and x86_64.

---

*All code written during the hackathon period (Aug 2026). Apache License 2.0.*
