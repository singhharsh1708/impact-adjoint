# Verification studies

Six studies that measure the machinery rather than any application. Each
writes an artifact that `experiments/collect_results.py` reads into
[results](results.md), so no number is retyped by hand.

## Solver and gradient

```{image} ../figures/study_verification.png
:alt: Three verification panels: observed order of accuracy 3.99 against the analytic solution, gradient-versus-finite-difference V-curves bottoming below 1e-8 on four probes, and VJP cost affine in parameter count at R-squared 0.998.
:width: 100%
```

```{figure-source} study_verification.png
```

**Convergence.** Order 3.99 on a smooth arc measured against the analytic drag
solution. The multi-bounce flat case sits at 10⁻¹² against the symbolic closed
form, where the floor is event localization rather than the integrator.

**Gradient accuracy.** Agreement with central differences traces the expected V
in the step size on all four probes, bottoming below 10⁻⁸. This is a
falsification test rather than a spot check: a wrong analytic gradient produces
a flat floor instead of a V, because the disagreement would be dominated by
the gradient error rather than by the step.

**Cost scaling.** Affine in parameter count out to 581 parameters, 93
microseconds per parameter, R² = 0.998. This is what makes the reverse-mode
extension a measured argument rather than a preference.

## Optimizer benchmark

A single run from one start point cannot support a claim about methods, so
this repeats the 24-dimensional design from five random starts with **both**
methods tuned over a grid per seed, learning rate for Adam and sigma0 for
CMA-ES.

```{image} ../figures/study_optimizers.png
:alt: Median convergence curves with interquartile bands over five random starts under both cost accountings: per evaluation Adam ends about 900 times below tuned CMA-ES, and the ranking reverses under measured wall-clock.
:width: 100%
```

```{figure-source} study_optimizers.png
```

Per evaluation, gradients win decisively: median final objective 3.4×10⁻⁷ for
Adam against 3.2×10⁻⁴ for tuned CMA-ES, a factor of about 900, with
Nelder-Mead three orders behind that. Charged by measured wall-clock, where
each gradient call costs a measured 6.8 forward solves, the ranking
reverses at this budget: Adam reaches 7.3e-04 and CMA-ES is roughly
2 times better.

:::{important}
The reversal is real and is reported rather than hidden. It is an
implementation property, not a fact about gradients. The VJP is
forward-variational, so it pays one variational column per parameter. A
reverse-mode saltation adjoint would return the same gradient for roughly the
cost of one solve, which would make the wall-clock panel resemble the
evaluation panel.
:::

What does not change under either accounting is that the gradient-free methods
never reach the design: at 24 dimensions they plateau three to five orders
above what Adam attains per evaluation.

## Robustness and generalization

```{image} ../figures/study_robustness.png
:alt: Held-out purity and separation margin for the point and ensemble designs across five independent ensembles, with non-overlapping Wilson intervals and a widened margin at the decision boundary.
:width: 100%
```

```{figure-source} study_robustness.png
```

**Purity with intervals.** Over five independent 200-particle ensembles the
point design classifies 983 of 1000 and the ensemble design 1000 of 1000, with
non-overlapping Wilson intervals. More useful than the headline percentage is
the separation margin: its fifth percentile improves from 0.05 m to 0.58 m
(bootstrap 95% interval [+0.48, +0.55] m) and the worst case moves from inside
the wrong bin to 0.09 m clear of the boundary. The median margin drops
slightly, because the ensemble objective trades the centre for the tail.

**Decisiveness under jitter.** Sweeping restitution with inlet jitter shows
what the deterministic sweep hid. The point design is indecisive at 2 of 20
restitutions, including its own trained value of 0.8, where 10% of jittered
draws cross into the wrong bin, and 45% at `e = 0.85`. The ensemble design is
unanimous at all 20.

That comparison is the argument for designing against an ensemble, and it
costs a little median margin to get it.
