# impact-adjoint: end-to-end gradients through impact events

**Track 1 — Inverse design & shape optimization** (cross-listed: Track 4,
differentiable inference). Solo entry.

**In one sentence:** we differentiate *through impacts* — exactly, not by
smoothing — and use it to design passive structures whose function only exists
because of those impacts: a bounce separator that sorts particles by material,
a terrain that routes different inlet speeds to different cups, and Bayesian
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

Experiment E3 makes this concrete, and includes its own steelman. A pure-JAX
simulation of a bouncing ball (RK4 scan, reset applied at the grid point via
`jnp.where`) converges to the correct *trajectory* as `dt → 0` — and reports
`d x(T)/d v0y = 0.0` at every resolution (an exact zero specific to
state-independent resets over flat terrain; on curved terrain the same program
returns nonzero *wrong* values instead). The true value is `+0.0904`, which
the closed-form oracle of Section 4 pins independently of this solver. The honest repair inside pure JAX —
interpolate the crossing time from the guard and reset at the interpolated
state — recovers a converging gradient, and that is precisely the point: the
repair *is* first-order event-time sensitivity machinery, hand-implemented,
with dt-dependent, non-monotone error (2×10⁻³ → 6×10⁻⁵ over our sweep) and a
new hand derivation owed for every guard/reset pair. The saltation endpoint
serves the exact value at every `dt` with no event handling in the consuming
program at all:

![E3](figures/e3_bias.png)

## 2. What we built

A two-Tesseract pipeline in which the event-aware machinery lives behind a
Tesseract boundary, and JAX never needs to know:

```
  ┌───────────────────────── one jax.grad, via tesseract-jax ─────────────────┐
  │                                                                          │
  ▼                                                                          │
(v0, e, mu, terrain θ)                                            optax Adam ┘
      │                                                                ▲
      ▼                                                                │
┌─────────────────────┐  qf, impact_x   ┌───────────────────┐   loss   │
│ contact-sim         │ ──────────────▶ │ score-target      │ ─────────┘
│ Julia · RK4 + events│                 │ JAX · autodiff    │
│ variational X +     │                 │ (landing          │
│ saltation jumps     │                 │  objective)       │
└─────────────────────┘                 └───────────────────┘
```

**contact-sim** (Julia): a 2D point mass under gravity (optional linear drag)
over smooth terrain `h(x)` built from configurable Gaussian bumps (3 in
E1/E4, 24 in E5). Guard `g(q) = y − h(x)`;
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

Cost model, stated plainly: the sensitivities are *forward*-variational, so a
VJP costs O(n_params): measured 6.0× a forward solve at 14 parameters and
8.6× at 77, milliseconds either way (apply 2.0 / 4.9 ms, VJP 12.1 / 41.6 ms
warm). The right trade at tens of design variables; at thousands, the
natural extension is a reverse-mode saltation adjoint behind the *same*
endpoint — the interface is already shaped for it.

**score-target** (JAX): a differentiable landing objective (quadratic distance
to a cup plus kinetic penalty), built with the `tesseract init --recipe jax`
endpoints.

### Why this needs Tesseract

The two components disagree about how to compute *and* how to differentiate:
dual-number forward AD plus analytic event handling in Julia, versus
reverse-mode tracing in JAX. Expressing the exact saltation update natively
inside `jax.grad` would mean reimplementing the solver as a custom primitive
with hand-written VJPs, which is a worse, unshareable Tesseract. The contract
is the seam: Julia publishes *what its derivatives are*, not *how it got
them*, and `tesseract-jax` splices them into JAX's chain rule.

A deliberate design point: when a trajectory leaves the model's validity
region (event capacity, Zeno chatter), the solver *terminates at the event
with an explicit status* and returns the **total derivative at the truncation
point**, event-time dependence included — so optimizers never consume
silently nonphysical states. Our adversarial testing showed the alternative
(integrate on, ignore events) produces plausible-looking gradients of a ball
in free-fall below the terrain.

## 3. Gradients doing real work

**E1 — inverse design.** Optimize launch velocity and restitution so the ball
lands in a cup 1.1 m past where the initial guess settles, every Adam step one
`jax.grad` through both Tesseracts ([trajectory](figures/e1_trajectory.png),
[convergence](figures/e1_convergence.png),
[animation](figures/e1_optimization.gif)). Miss distance falls **1.12 m →
2.7 cm** through five impacts, with the optimizer repeatedly crossing
bounce-count boundaries on the way (4 → 6 → 5, with excursions to 8). The
objective is discontinuous at those crossings, which is inherent to contact;
the fixed-topology gradients on each side are exact, and that is what carries
Adam through.

**E2/E2b — calibration (Track 4).** Recover material parameters `(e, μ)`
from the *positions of three impacts* observed with 5 mm noise — an observable
that exists only because of events. Point estimation via the solver's VJP
recovers `e` to 0.002 and `μ` to 0.009 from a distant start; the full
posterior: NumPyro's NUTS sampler, whose Hamiltonian dynamics require a
JAX-differentiable log-density, runs directly against the *containerized*
solver — every leapfrog step calls the Tesseract's apply and saltation-VJP
endpoints over HTTP — ~10,000 solver calls per chain. Two chains, zero
divergences, r̂ = 1.01: posterior `e = 0.697 ± 0.007`, `μ = 0.096 ± 0.009`,
with the truth inside the 68% and 95% credible intervals of both marginals,
and the
posterior resolves the physically meaningful e–μ ridge, more bounce traded
against more tangential loss ([figure](figures/e2b_posterior.png)).

This is the composition Track 4 asks for: an expensive event-driven solver
dropped into a probabilistic workflow unchanged. The container is load-bearing
rather than packaging, since NumPyro's jitted callbacks run off the main thread
and an in-process embedded Julia runtime does not tolerate that, while over
HTTP the problem does not arise (reported upstream as tesseract-jax#234). The
same solver is also driven with nothing but `curl`
(`scripts/second_client_curl.sh`).

**E4 — terrain design.** The terrain parameters are differentiable inputs, so
the *structure* can be the design variable: one terrain routes balls entering
at 1.6 and 2.6 m/s to two different cups (miss 2.88 m / 0.48 m falling to
**2.2 / 3.0 cm**); the optimizer flattens two bumps and keeps one narrow
deflector the slow ball cannot clear ([figure](figures/e4_sorter.png)).

**E5 — bounce separator, 24-dimensional, head-to-head (the headline).**
Industrial bounce/impact separators sort particles by resilience. We design
one: two particles enter *identically* and differ only in restitution
(`e = 0.5` "rubber" vs `e = 0.8` "PET"); the design vector is the 24 bump
amplitudes of the surface; each particle must land in its own bin.

![E5](figures/e5_separator.png)

The designed surface separates the materials from a shared first impact —
rubber dies into bin A within 8 bounces, PET carries over the designed hills
into bin B (the optimizer grew a backstop that bounces PET *backward* into its
bin) — with landing errors of 0.38 mm and 0.34 mm. The right panel makes
"gradients doing real work" quantitative, under both accountings.
Charging a gradient call as two forward solves, Adam reaches **2×10⁻⁷** while
CMA-ES plateaus at 2×10⁻³ (~8,800× worse) and Nelder-Mead at 2×10⁻²; a
3-seed × 3-σ₀ CMA tuning grid (`experiments/e5_cma_grid.py`) gives median
1.7×10⁻³ with a best tail of 1.7×10⁻⁴ — still ~750× above Adam. Charging by *measured wall-clock* (a VJP costs 8.6
forward solves at these 77 parameters), Adam at CMA's total wall-clock sits in
CMA's own range (~4×10⁻³); the separation opens beyond that point, where Adam
descends four further orders. The gradient-free runs are not equally stuck:
CMA-ES is flat over its last third, while Nelder-Mead is still improving when
the budget ends (its last gain lands at evaluation 896 of 900), so its 2×10⁻²
is a budget limit rather than a converged value. Neither is within three
orders of the gradient result. In 24 dimensions, exact gradients through the
events are the difference between solving this design problem and not.

**E5b — design under uncertainty.** Real separators process streams with
scatter, so we make the expected loss over a particle ensemble (inlet velocity
sd 5 cm/s, per-particle restitution sd 0.03, fixed common random numbers — the
ensemble gradient is exact) the design objective. Two results: the E5 point
design is already robust, scoring **99.5%** sorting purity on 200 held-out
particles; and warm-started ensemble refinement reaches **100%** held-out
purity while visibly widening the separation margin between the two materials'
landing distributions ([figure](figures/e5b_purity.png)).

**E6 — zero-shot generalization (the surprise).** The E5 surface was
optimized for exactly two restitution values. Sweeping the continuum it never
saw, the designed geometry acts as a *classifier*: every material with
`e ∈ [0.35, 0.875]` is binned by a single threshold (A up to `e = 0.65`, B
from `e = 0.675`). Margins are asymmetric and worth stating: the trained
`e = 0.5` point lands 0.80 m clear of the decision boundary, the trained
`e = 0.8` point only 0.13 m. Under 3 cm/s of inlet-velocity scatter at
off-design materials, `e = 0.45` classifies 12 of 12 correctly and `e = 0.85`
10 of 12. The failure edge is physical and we report it: beyond `e ≈ 0.9` classification
stops being clean. At `e = 0.9` the ball rebounds off the backstop and exits
left into the wrong bin; at `e = 0.925` it overshoots 1.2 m past bin B; at
`e = 0.95` it lands between the bins on a capacity-truncated trajectory.
Nothing in
the objective asked for any of this — the generalization emerged from
optimizing two point designs through their impact events:

![E6](figures/e6_generalization.png)

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

Near tangency the impact sensitivities grow as `δ^(−1/2)` (the saltation
denominator `g_q · f⁻` vanishes at grazing) — a property of the physics, not
an artifact; the solver raises an explicit error at degenerate crossings
rather than returning an unbounded gradient.

## 5. Related work: how everyone else gets contact gradients

Existing engines obtain contact gradients by AD through smoothed/penalty
stepping (Brax, NeurIPS 2021; DiffTaichi, ICLR 2020; gradSim, ICLR 2021), by
implicit differentiation of a relaxed complementarity solve (Dojo, arXiv 2022;
Nimble, RSS 2021), or from learned contact models (Zhong et al., NeurIPS
2021) — each trading event-gradient fidelity for what the host framework can
express; Suh et al. (ICML 2022) document the resulting bias. The exact
alternative is classical: the **saltation matrix** (Aizerman & Gantmacher
1958; Hiskens & Pai, IEEE TCAS 2000; surveyed by Kong et al., Proc. IEEE
2024). Our contribution is not the formula but the *packaging*: exact event
sensitivities behind Tesseract endpoints, consumable by frameworks that cannot
express them natively. The closest JAX-native alternative is Diffrax (Kidger,
2021), which localizes events by root finding; the difference is where the
event sensitivity is authored, and for a solver that already exists in another
language the component boundary avoids porting it at all. To our knowledge no prior work optimizes the
*environment/structure* through contact events (nearest: robot morphology
design, Xu et al., RSS 2021; soft-body design, Geilinger et al., TOG 2020) —
the terrain-as-design-variable problems appear new.

## 6. Limitations

- Impact law: kinematic Newton restitution + tangential retention, not a
  Coulomb cone; sticking/sliding modes terminate explicitly (`status=2`).
- Gradients are exact at fixed event topology; objectives are discontinuous
  across bounce-count boundaries (physical; visible in E1).
- Event detection resolves features wider than `|vx|·dt/3` (documented on the
  schema); 2D single body — a deliberate trade for verified correctness
  within the hackathon period.

## 7. Reproducibility

Everything reproduces with the README commands, CPU-only, in minutes; images
build for arm64 and x86_64; CI runs the full validation chain on every push.
An engineering finding from this work is filed upstream
([tesseract-jax#234](https://github.com/pasteurlabs/tesseract-jax/issues/234)).

---

*All code written during the hackathon period (Aug 2026). Apache License 2.0.*
