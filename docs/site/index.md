# impact-adjoint

**Exact gradients through impact events, across a Julia and JAX Tesseract
boundary.**

Simulate a bouncing ball in JAX the natural way, applying the impact at the
integrator step where it is detected, and `jax.grad` returns
`d x(T)/d v0y = 0.0` exactly on flat terrain, when the truth is `+0.09`.
impact-adjoint fixes this exactly, using classical saltation-matrix event
sensitivities served from a Julia solver through a Tesseract boundary, and
uses the gradients to design passive structures whose function only exists
because of impacts.

```{image} ../figures/e5_learning.gif
:alt: the designed surface learns to sort two materials
:width: 100%
```

*One terrain, two materials, same inlet. Gradient descent through every impact
of both trajectories reshapes the surface until restitution alone routes each
particle to its own bin.*

::::{grid} 3
:gutter: 2

:::{grid-item-card} The problem
:link: problem
:link-type: doc

Autodiff through a time-stepping simulator returns confident, finite, wrong
gradients the moment the dynamics have events.
:::

:::{grid-item-card} The method
:link: method
:link-type: doc

Saltation-matrix event sensitivities in Julia, published through Tesseract
endpoints, spliced into JAX's chain rule.
:::

:::{grid-item-card} The results
:link: results
:link-type: doc

Seven experiments and six verification studies, every number generated from
committed artifacts.
:::
::::

## At a glance

| | |
|---|---|
| Track | 1, inverse design and shape optimization |
| Components | `contact-sim` (Julia), `score-target` (JAX), composed under one `jax.grad` |
| Solver accuracy | order 3.99; 10⁻¹² against a symbolic closed form |
| Gradient accuracy | agrees with finite differences to below 10⁻⁸; 0 failures in 1574 `check-gradients` checks |
| Headline design | a 24-parameter surface that sorts particles by restitution alone |
| Reproduction | about 5 minutes, CPU only |

## Where to start

- New here: read [the problem](problem.md), then [the method](method.md).
- Want the evidence: [experiments](experiments.md) and [verification studies](studies.md).
- Want to run it: [getting started](getting-started.md).
- Want the numbers: [results](results.md), generated from the artifacts.

```{toctree}
:hidden:
:maxdepth: 2

problem
method
getting-started
experiments
studies
results
reference
upstream
limitations
```
