# impact-adjoint

[![CI](https://github.com/singhharsh1708/impact-adjoint/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/singhharsh1708/impact-adjoint/actions/workflows/test.yaml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Simulate a bouncing ball in JAX the natural way, applying the impact at
the integrator step where it is detected, and `jax.grad` returns `d x(T)/d
v0y = 0.0` exactly on flat terrain, when the truth is `+0.09`.**
impact-adjoint fixes this exactly, using classical saltation-matrix event
sensitivities served from a Julia solver through a Tesseract boundary. The
gradients then design passive structures whose function only exists because
of impacts.

![The surface learns to sort](docs/figures/e5_learning.gif)
*One terrain, two materials, same inlet: gradient descent through every
impact of both trajectories reshapes the surface until restitution alone
routes each particle to its own bin.*

**Tesseract Hackathon 2026, Track 1: inverse design & shape optimization.**
(The Bayesian calibration and design-under-uncertainty experiments also touch
Track 4, but Track 1 is the entry's track.)

What the gradients design and infer:

- a **24-parameter resilience separator** that sorts particles by how far
  they bounce (the documented industrial cousins are seed resilience
  separators and potato/stone bounce rollers),
- **terrain** that routes different inlet speeds to different cups,
- **Bayesian recovery of material parameters** from where things actually hit.

## Contents

- [Architecture](#architecture) · [Results](#results) · [Correctness](#correctness)
- [Performance envelope](#performance-envelope) · [Reproduce](#reproduce) · [Troubleshooting](#troubleshooting)
- [Repository structure](#repository-structure) · [Limitations](#model-and-scope-honest-limitations) · [Future work](#future-work)
- [Upstream fixes](#upstream-fixes-from-this-work) · [References](#references)

## Architecture

```mermaid
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

The two components *disagree about how to differentiate*. One side uses
dual-number forward AD plus analytic event handling in a Julia process, the
other reverse-mode tracing in JAX, and `tesseract-jax` composes them into a
single `jax.grad` anyway. The same solver container also serves NumPyro's HMC over HTTP and a
raw `curl` client, unchanged.

- **`tesseracts/contact_sim`** (Julia). 2D ballistic flight over configurable
  Gaussian-bump terrain with impact events (restitution `e`, tangential loss
  `mu`); all differentiable Tesseract endpoints, gradients from forward
  variational equations with analytic **saltation-matrix** updates at events.
- **`tesseracts/score_target`** (JAX). Differentiable landing objective.
- (`tesseracts/julia_kernel` is the minimal Day-1 boundary proof, driven by
  `scripts/proof_local.py` / `scripts/proof_container.py`.)

## Results

| Experiment | Headline |
|---|---|
| **E3**, the failure measured | Grid-reset autodiff gives `d x(T)/d v0y` = **exactly 0.0 at every dt** (truth +0.0904; the exact zero is specific to this flat-terrain case, on curved terrain it is nonzero and wrong). The pure-JAX repair converges only by hand-implementing event sensitivity. |
| **E5**, 24-dim resilience separator | At a shared budget of 900 forward-solve units with a gradient charged as 2: Adam on saltation gradients **2×10⁻⁷** vs CMA-ES 2×10⁻³ vs Nelder-Mead 2×10⁻². Charged by measured wall-clock instead, Adam loses at CMA's own budget and wins only by continuing four orders further; the writeup gives both accountings in full. |
| **E5b**, design under uncertainty | Ensemble objective over inlet and restitution scatter. Held-out sorting purity 199/200 for the point design, 200/200 after refinement, with a visibly wider margin at the decision boundary. |
| **E6**, zero-shot generalization | Trained on two materials, sorts the whole continuum e ∈ [0.35, 0.875] with **one threshold**. |
| **E1**, inverse design | Miss **1.12 m → 2.7 cm** through 5 bounces, across bounce-count changes. |
| **E2/E2b**, calibration | NUTS posterior `e = 0.697 ± 0.007`, `mu = 0.096 ± 0.009`; truth inside both 95% CIs, 0 divergences. |
| **E4**, terrain design | One terrain routes two inlet speeds to two cups (miss 2.2 / 3.0 cm). |

![E3](docs/figures/e3_bias.png)
*The thesis in one figure: the natural autodiff program returns exactly zero
at every step size (left); the pure-JAX repair converges erratically, and
only the saltation endpoint is exact at any dt (right).*

![E5 separator](docs/figures/e5_separator.png)
*The designed separator (left) and the 24-dimensional head-to-head (right):
at equal budget the gradient-free runs end three to five orders above Adam.*

## Correctness

The gradients are validated against **solver-independent oracles** as well
as finite-difference gates through the solver itself:

- `scripts/validate_closed_form.py` runs the full multi-bounce closed form on flat
  terrain, derived in sympy rational arithmetic (including hand-differentiated
  impact recursions): Jacobian agreement **7e-12**.
- `scripts/validate_reference.py` is an independent reimplementation of the spec
  (scipy RK45 + its own event localization): primal agreement 1e-10; analytic
  Jacobian vs finite differences *through the independent implementation*
  **5e-9**, covering the sloped-contact-frame and drag sectors.
- `tesseract run contact-sim check-gradients` is Tesseract's built-in checker:
  **0 failures / 1574 checks** per gradient endpoint.
- `scripts/validate_contact.py` is an FD gate (rtol 1e-5) plus robustness
  regressions: chatter/settle termination, event-capacity truncation,
  sub-step-width terrain features, energy conservation at `e=1` (relative
  drift 5e-13), and `jax.grad` end-to-end.

Trajectories that leave the model's validity region **terminate with an
explicit status** (event capacity / settled contact) and still return
well-defined total-derivative Jacobians at the truncation point, so
optimization loops never consume silently nonphysical states.

## Verification studies

Beyond the correctness oracles above, four studies measure the machinery
itself. Every number is collected into [docs/RESULTS.md](docs/RESULTS.md) by
`experiments/collect_results.py`, read from the committed artifacts rather
than retyped.

![verification](docs/figures/study_verification.png)
*Order 3.99 on a smooth arc against the analytic solution; gradient-vs-FD
V-curves bottoming below 1e-8 on four probes; cost affine in parameter count
at R2 = 0.998.*

![benchmark](docs/figures/study_optimizers.png)
*Five random starts, both methods tuned per seed. Per evaluation Adam ends
about 900x below tuned CMA-ES. Charged at measured wall-clock, where each
gradient costs 8.5 solves, CMA-ES is about 24x better at this budget. Both
are reported; the reverse-mode adjoint in Future work is what closes it.*

![robustness](docs/figures/study_robustness.png)
*Five independent ensembles: 983/1000 vs 1000/1000 with non-overlapping
Wilson intervals, and the fifth-percentile margin improving 0.05 m to
0.58 m. Under inlet jitter the point design is indecisive at 5 of 20
restitutions, including its own trained value; the ensemble design at none.*

## Performance envelope

Warm per-call cost of the solver component (M-series CPU, dev mode;
container adds ~10 ms HTTP overhead per call):

| call | 3 bumps, 14 params (dt 1e-3, t 2.0 s) | 24 bumps, 77 params (dt 5e-4, t 2.2 s) |
|---|---|---|
| `apply` | 2.0 ms | 4.9 ms |
| `vector_jacobian_product` | 12.1 ms | 41.6 ms |
| ratio | 6.0× | 8.5× |

Gradients are forward-variational, so VJP cost grows with parameter count.
HMC through the container sustains roughly 10,000 endpoint calls (apply plus
VJP pairs) in about 2.5 minutes of sampling.

## Reproduce

> [!TIP]
> **Figures in one minute:** all experiment results are committed as
> `experiments/*.npz` / `*.npy`, so
> `python experiments/make_figures.py && python experiments/make_e5_figure.py && python experiments/make_e5b_figure.py`
> regenerates all eight static figures without rerunning any optimization
> (after the one-time Julia bootstrap). The two GIFs come from
> `make_animation.py` / `make_e5_animation.py`, which do re-run their loops.

Requires Python ≥ 3.12, Docker (for containerized runs), and ~5 GB disk for
the Docker images (contact-sim ~3.8 GB, score-target ~1.3 GB). Julia itself
is bootstrapped automatically by `juliacall`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install "tesseract-core[runtime]==1.11.0" tesseract-jax==0.4.1 "jax==0.11.0" \
            optax equinox juliacall==0.9.31 numpy scipy sympy matplotlib numpyro pytest cma

# validation (dev mode, no Docker needed)
python scripts/proof_local.py              # 5 s boundary proof
python scripts/validate_contact.py
python scripts/validate_closed_form.py
python scripts/validate_reference.py
pytest tests/

# experiments (dev mode)
python experiments/e3_naive_vs_saltation.py
python experiments/e1_inverse_design.py
python experiments/e2_calibration.py
python experiments/e4_terrain_design.py
python experiments/e5_separator.py          # ~30 s warm: 3 optimizers x 900 evals
python experiments/e5b_robust_separator.py  # ~10 min: ensemble design + purity eval
python experiments/e6_generalization.py

# verification studies (write the artifacts behind docs/RESULTS.md)
python experiments/study_convergence.py
python experiments/study_gradient_accuracy.py
python experiments/study_scaling.py
python experiments/study_optimizers.py          # ~25 min: 5 seeds x tuned grids
python experiments/study_robustness_stats.py    # ~10 min
python experiments/study_generalization_stats.py  # ~15 min
python experiments/make_study_figures.py
python experiments/collect_results.py

# containerized (any docker-compatible engine; needs buildx; ~10 min for contact-sim)
tesseract build tesseracts/contact_sim
tesseract build tesseracts/score_target
tesseract run contact-sim check-gradients @tesseracts/contact_sim/check_payload.json
python experiments/e2b_bayesian.py                   # NUTS, 2 chains: ~15 min
python experiments/e1_inverse_design.py --container  # same optimization via the served images
./scripts/second_client_curl.sh                      # gradients with nothing but curl
```

> [!NOTE]
> The first Julia call bootstraps a project environment (and Julia itself if
> absent), so expect 1 to 5 minutes and a wall of `[juliapkg]` output the first
> time; warm runs take seconds. Measured budget: about 5 minutes for the
> validation block and experiments E1 to E6, plus ~10 minutes for
> `e5b_robust_separator.py`, plus ~10 minutes to build the contact-sim image
> and ~15 minutes for the two NUTS chains.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `tesseract` runs an OCR tool | PATH collision with Tesseract-OCR; use the venv's `tesseract` binary. |
| Wall of `[juliapkg] Installing...` output | One-time Julia bootstrap; subsequent runs are warm. |
| `docker build` fails with `unknown flag: --load` | Docker buildx plugin missing (common with colima/podman setups). |
| Container cannot write outputs (colima/VM setups) | Pass an `output_path` under your home directory; VM file sharing may not cover system temp dirs. |

## Repository structure

```
tesseracts/    contact_sim (Julia solver) · score_target (JAX objective) · julia_kernel (Day-1 proof)
experiments/   e1–e6, e5b + figure/animation generators + committed result artifacts
scripts/       three validation oracles · boundary proofs · curl client
tests/         12 golden regression tests (run in CI)
docs/          technical writeup + all figures
```

## Model and scope (honest limitations)

- Impact law: kinematic Newton restitution plus a tangential retention
  factor, not a full Coulomb friction cone; sticking/sliding contact modes are out of
  scope and trajectories entering them terminate with `status=2`.
- Gradients are exact for the continuous hybrid system at fixed event topology;
  across bounce-count boundaries the objective is discontinuous (inherent to
  the physics, visible and handled in E1).
- Event detection resolves terrain features wider than `|vx|·dt/3`
  (documented per-input); grazing impacts have an inherent `δ^(-1/2)`
  sensitivity growth near tangency.

## Future work

- **Reverse-mode saltation adjoint.** The sensitivities here are forward
  variational, so VJP cost scales with parameter count. A backward costate
  integration with `Sᵀ` jumps at events would serve thousands of design
  variables behind the same endpoint, with no change for the client.
- **Coulomb friction cone.** The impulse-ratio (Routh) reset is a ~4-line
  change to `reset_map`, at the cost of re-deriving the closed-form oracle;
  it would extend the model into sticking and sliding contact.
- **3D and multi-body.** The saltation machinery is dimension-agnostic; the
  work is in guard geometry and event bookkeeping, not the sensitivities.
- **Upstream.** Test the tesseract-jax fix for
  [#234](https://github.com/pasteurlabs/tesseract-jax/issues/234) against this
  solver once it lands.

See [docs/writeup.md](docs/writeup.md) for the technical writeup.

## References

- Aizerman & Gantmacher (1958); Hiskens & Pai, *IEEE TCAS* (2000); Kong,
  Payne, Zhu & Johnson, *Proc. IEEE* (2024). The saltation-matrix lineage.
- Suh, Simchowitz, Zhang & Tedrake, *ICML* (2022), on the bias of simulator
  gradients through contact.
- [Tesseract Core](https://github.com/pasteurlabs/tesseract-core) ·
  [Tesseract-JAX](https://github.com/pasteurlabs/tesseract-jax) provide the
  component boundary this entry is built on.

## Upstream fixes from this work

Problems found while building this, reported and fixed upstream during the
hackathon period:

| Where | What |
|---|---|
| [core#666](https://github.com/pasteurlabs/tesseract-core/issues/666) / [PR #667](https://github.com/pasteurlabs/tesseract-core/pull/667) | VJP cache compared keys by hash alone, so a collision silently served another input's gradient. Reproduced on the shipped `vectoradd_jax` example. |
| [jax#235](https://github.com/pasteurlabs/tesseract-jax/issues/235) / [PR #236](https://github.com/pasteurlabs/tesseract-jax/pull/236) | `jax.jvp` returned another element's tangent for partially-differentiated list inputs, disagreeing with `jax.grad` silently. |
| [jax#234](https://github.com/pasteurlabs/tesseract-jax/issues/234) | Jitted callbacks deadlock Tesseracts embedding an in-process Julia runtime ([reproducer](https://github.com/singhharsh1708/tesseract-jax-234-repro)). |
| [mosaic#126](https://github.com/pasteurlabs/mosaic/pull/126), [#141](https://github.com/pasteurlabs/mosaic/pull/141) | Harness fixes: Docker setup diagnostics, anomaly status reporting. |

## Author

Harsh Singh ([@singhharsh1708](https://github.com/singhharsh1708)), solo
entry. Questions: open an issue on this repository.

## License

Apache License 2.0.
