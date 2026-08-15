# impact-adjoint

Exact gradients through impact events, across a Julia and JAX Tesseract
boundary.

```{raw} html
<div class="ia-claim">
Simulate a bouncing ball in JAX the natural way, applying the impact at the
integrator step where it is detected, and <code>jax.grad</code> returns
<strong class="ia-zero">d x(T)/d v0y = 0.0</strong> exactly on flat terrain.
The true value is <strong class="ia-zero">+0.09</strong>.
</div>
```

The gradient is not noisy or approximate. It is confidently, silently wrong,
and refining the step size does not help, because the step at which the event
fires is an integer and autodiff cannot differentiate it. impact-adjoint
computes the missing term exactly, using the classical saltation matrix, and
serves it from a Julia solver through a Tesseract boundary so that JAX can
consume it without knowing any of this.

The gradients then design passive structures whose function only exists
because of impacts.

```{figure} ../figures/e5_learning.gif
:alt: a designed surface learning to sort two materials by restitution
:width: 100%
:name: fig-sorter

One terrain, two materials, same inlet. Gradient descent through every impact
of both trajectories reshapes the surface until restitution alone routes each
particle to its own bin.
```

```{raw} html
<dl class="ia-figures">
  <div><dt>3.99</dt><dd>observed order of accuracy, against an analytic solution</dd></div>
  <div><dt>10<sup>-12</sup></dt><dd>multi-bounce agreement with a symbolic closed form</dd></div>
  <div><dt>0 / 1574</dt><dd>failures in Tesseract's own gradient checker</dd></div>
  <div><dt>24</dt><dd>design variables in the headline separator</dd></div>
  <div><dt>~5 min</dt><dd>to reproduce from a cold clone</dd></div>
</dl>
```

## Where to start

::::{grid} 3
:gutter: 2

:::{grid-item-card} The problem
:link: problem
:link-type: doc

Why autodiff returns a wrong gradient at an impact, what the pure-JAX repair
costs, and what Diffrax can and cannot do about it today.
:::

:::{grid-item-card} The method
:link: method
:link-type: doc

The saltation jump condition, the component boundary that carries it, and the
termination semantics that keep optimizers honest.
:::

:::{grid-item-card} The evidence
:link: studies
:link-type: doc

Six verification studies and seven experiments, every number generated from a
committed artifact.
:::

::::

## At a glance

```{list-table}
:header-rows: 1
:widths: 26 74

* - Item
  - Detail
* - Track
  - 1, inverse design and shape optimization
* - Components
  - `contact-sim` (Julia) and `score-target` (JAX), composed under one `jax.grad`
* - Gradient method
  - forward variational sensitivities with analytic saltation updates at events
* - Headline result
  - a 24-parameter surface that sorts particles by restitution alone
* - Validation
  - two solver-independent oracles, twelve golden tests, CI on every push
```

```{toctree}
:hidden:
:caption: Orientation

problem
method
getting-started
```

```{toctree}
:hidden:
:caption: Evidence

experiments
studies
results
```

```{toctree}
:hidden:
:caption: Reference

reference
upstream
limitations
```
