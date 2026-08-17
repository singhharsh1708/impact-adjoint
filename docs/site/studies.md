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
:alt: Median convergence curves with interquartile bands over five random starts under both cost accountings: per evaluation Adam ends about 347 times below tuned CMA-ES on the paired per-seed median, and under measured wall-clock the ordering is not resolved at five seeds.
:width: 100%
```

```{figure-source} study_optimizers.png
```

Per evaluation, gradients win decisively: median final objective 3.4×10⁻⁷ for
Adam against 3.2×10⁻⁴ for tuned CMA-ES. The seeds are paired, so the honest
statistic is the median of the per-seed ratios, 347x, rather than the
ratio of the medians; and the per-seed ratios span 12x to 139843x, so
the direction is unanimous across five seeds but the magnitude is not
resolvable at this sample size. With
Nelder-Mead three orders behind that. Charged by measured wall-clock, where
each gradient call costs a measured 6.8 forward solves, the ranking
reverses at this budget: Adam reaches 7.3e-04 and CMA-ES is roughly
better on the ratio of medians. That reversal is not resolved at this sample
size: CMA-ES is ahead on 4 of 5 seeds and behind on 1, a sign test gives
p = 0.375, and a bootstrap interval on the median per-seed ratio spans both
sides of parity. The honest statement is that under wall-clock accounting the
ordering is not established at n = 5, not that it reverses.

:::{important}
The reversal is real and is reported rather than hidden. It is an
implementation property, not a fact about gradients. The VJP is
forward-variational, so it pays one variational column per parameter. A
reverse-mode saltation adjoint would return the same gradient for roughly the
cost of one solve, which would make the wall-clock panel resemble the
evaluation panel.
:::

What does not change under either accounting is the objective value: at 24
dimensions the gradient-free methods plateau 1.1 to 5.1 orders above what Adam
attains per evaluation.

:::{important}
**That gap does not survive translation into engineering units, and we checked
rather than assumed.** The loss is squared landing error summed over two
particles, so Adam's 2.25e-07 is a 0.34 mm miss and tuned CMA-ES's 3.2e-04 is
12.6 mm, against bins 1600 mm apart. Scoring all four designs on the held-out
scatter ensemble that E5b uses:

| design | purity | 5th-percentile margin |
|---|---|---|
| Adam | 0.995 | +0.07 m |
| CMA-ES | 0.980 | +0.18 m |
| Nelder-Mead | 0.995 | +0.41 m |
| ensemble-refined | 1.000 | +0.43 m |

Nelder-Mead is five orders behind on the objective and sorts exactly as well
as Adam, with a *wider* margin. Minimising the point objective further does
not buy a better separator at this scale. What buys one is optimising the
ensemble objective, which needs a gradient through many trajectories at once,
and that is where the gradient actually pays here, not in the last four orders
of a point fit.
:::

## Robustness and generalization

```{image} ../figures/study_robustness.png
:alt: Held-out purity and separation margin for the point and ensemble designs across five independent ensembles, with non-overlapping Wilson intervals and a widened margin at the decision boundary.
:width: 100%
```

```{figure-source} study_robustness.png
```

**Purity with intervals.** Over five independent 200-particle ensembles the
point design classifies 983 of 1000 and the ensemble design 997 of 1000, with
non-overlapping Wilson intervals. More useful than the headline percentage is
the separation margin: its fifth percentile improves from 0.05 m to 0.49 m
(paired bootstrap 95% interval [+0.39, +0.47] m). The worst case does **not**
improve: it goes from -0.12 m to -0.35 m, both inside the wrong bin. The
ensemble objective buys the low tail, not the extreme, and the median margin
drops slightly because the centre is what it trades away. A change in the
median is not distinguishable from zero at this sample size, so we do not
claim one.

**Decisiveness under jitter.** Sweeping restitution with inlet jitter shows
what the deterministic sweep hid. The point design is indecisive at 2 of 20
restitutions, including its own trained value of 0.8, where 12.5% of jittered
draws cross into the wrong bin. The ensemble design is
unanimous at all 20.

That comparison is the argument for designing against an ensemble, and it
costs a little median margin to get it.
