# Verification studies

{{ repo_note }}

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
extension a measured argument rather than a preference. The slope is
wall-clock on one shared laptop and moves between runs: a repeat measured 81
microseconds per parameter at R² = 0.997. The affine shape is what holds.

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
resolvable at this sample size. Nelder-Mead sits a further 3.5 orders behind
CMA-ES. Charged by measured wall-clock, where each gradient call costs a
measured 6.8 forward solves, the ordering at this budget is no longer in
Adam's favour: Adam reaches 7.3e-04 against CMA-ES at 3.2e-04, which is 2.3x
on the ratio of medians and 6.3x on the paired per-seed median. That is not
resolved at this sample size: CMA-ES is ahead on 4 of 5 seeds and behind on 1, a sign test gives
p = 0.375, and a bootstrap interval on the median per-seed ratio spans both
sides of parity. The honest statement is that under wall-clock accounting the
ordering is not established at n = 5, not that it reverses. The charge is
itself a wall-clock measurement on a shared machine, and the scaling study it
rests on moves about 11% between runs, which is one more reason to read this
as unresolved rather than as a result.

:::{important}
The wall-clock accounting is reported rather than hidden. What it exposes is
an implementation property, not a fact about gradients. The VJP is
forward-variational, so it pays one variational column per parameter. A
reverse-mode saltation adjoint would return the same gradient for roughly the
cost of one solve, which would make the wall-clock panel resemble the
evaluation panel.
:::

What does not change under either accounting is the objective value: at 24
dimensions the gradient-free methods plateau 1.1 to 8.0 orders above what Adam
attains per evaluation.

:::{important}
**That gap does not survive translation into engineering units, and we checked
rather than assumed.** The loss is squared landing error summed over two
particles, so Adam's 2.25e-07 is a 0.34 mm miss, while the CMA-ES design that
E5b actually scores sits at 1.98e-03, about 31 mm per particle. The bins are
1600 mm apart. Scoring all four designs on the held-out
scatter ensemble that E5b uses:

| design | purity (95% Wilson) | 5th-percentile margin, m (95% bootstrap) |
|---|---|---|
| Adam | 199/200 (97–100%) | +0.07 (+0.06 to +0.32) |
| CMA-ES | 196/200 (95–99%) | +0.18 (+0.11 to +0.25) |
| Nelder-Mead | 199/200 (97–100%) | +0.41 (+0.33 to +0.52) |
| ensemble-refined | 200/200 (98–100%) | +0.43 (+0.40 to +0.52) |

All four are scored on the same 200 particles, so the columns are paired.
The intervals come from `experiments/study_design_table.py` into
`design_table.json`: Wilson on the purities, and 10,000 bootstrap
resamples at a pinned seed on the fifth percentile.
The purity intervals overlap for every pair, so purity does not separate
these designs at this sample size; the margin intervals separate Adam from
both Nelder-Mead and the ensemble design, and do not separate those two
from each other.

```{image} ../figures/design_comparison.png
:alt: Left, the margin distribution each design produces on the held-out scatter ensemble as a cumulative curve; right, final objective against fifth-percentile margin, showing four orders of objective buying almost no separation.
:width: 100%
```

```{figure-source} design_comparison.png
```


Nelder-Mead is five orders behind on the objective and sorts exactly as well
as Adam, with a *wider* margin. Minimising the point objective further does
not buy a better separator at this scale. Optimising the ensemble objective
does separate from Adam, whose margin interval it clears. It does not separate
from Nelder-Mead on the margin: those two intervals overlap, and on this
ensemble the honest claim stops there.

We used to say the ensemble objective was where the gradient pays, because it
needs a gradient through many trajectories at once. That was asserted, and when
we finally measured it the claim did not survive. See
[the control](#the-control-gradient-free-on-the-ensemble-objective) below: what
buys the robustness is optimising the ensemble objective at all, not the
gradient that does it.
:::

## The control: gradient-free on the ensemble objective

For most of this project the answer to "Nelder-Mead matches Adam on held-out
purity" was that the ensemble objective is different, and that it needs a
gradient through many trajectories at once. Nobody had run a gradient-free
method on the ensemble objective, so the sentence was an assertion. We ran it.

Same warm start as E5b, same training draw, same held-out ensemble, same
objective. The budget is matched in particle solves: an Adam step issues an
`apply` and a `vector_jacobian_product` per particle, and the measured
gradient charge is the reverse pass alone, so a step costs `(1 + charge)` per
particle. That is {{ CTRL_budget_solves }} solves, or {{ CTRL_max_evals }}
gradient-free ensemble evaluations. CMA-ES was swept over three initial step
sizes and three seeds, because `e5_cma_grid.py` had already measured a 24x
spread across that grid and one hardcoded configuration would have decided the
result by the protocol. Nelder-Mead got an explicit initial simplex at two
scales, because scipy's default builds it as `x_i * 1.05`, which from this warm
start spans four orders of magnitude and freezes the small amplitudes.

| method | ensemble objective | held-out | 5th-percentile margin |
|---|---|---|---|
| warm start (E5 point design) | 7.3e-2 | 199/200 | +0.07 m |
| Adam, published `lr = 0.004` | {{ CTRL_adam_loss }} | 200/200 | +{{ CTRL_adam_p5 }} m |
| Adam, best of {{ SWEEP_n_lrs }} learning rates | {{ SWEEP_best_loss }} | 200/200 | +{{ SWEEP_best_p5 }} m |
| **CMA-ES, best of {{ CTRL_cma_grid_n }}** | **{{ CTRL_cma_loss }}** | 200/200 | **+{{ CTRL_cma_p5 }} m** |
| Nelder-Mead, best of 2 | {{ CTRL_nm_loss }} | {{ CTRL_nm_correct }} | {{ CTRL_nm_p5 }} m |

:::{important}
The claim did not survive. CMA-ES reaches a better ensemble design than Adam at
a matched budget, on the objective and on the tail, and it does so on every one
of its {{ CTRL_cma_grid_n }} grid runs; its median, {{ CTRL_cma_median_loss }},
still beats Adam's best. Because the first comparison gave CMA-ES nine
configurations and Adam one, we then swept Adam's learning rate over
{{ SWEEP_n_lrs }} values fixed before the run. It does not change the ordering,
and it turns up something else: the published `lr = 0.004` is not Adam's best
here. `lr = 0.001` reaches {{ SWEEP_best_loss }}.

So the honest statement is that optimising the ensemble objective is what buys
the robustness, and the gradient is not what makes that possible. Nelder-Mead
is genuinely worse, so this is not "gradient-free wins"; it is CMA-ES being
well suited to a 24-dimensional box with a good warm start.

What remains true of the gradient is cost per step, not reachability, and this
project has not measured that cleanly enough to lead with it.
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
ensemble objective buys the low tail, not the extreme, and it pays in the
centre: the median margin drops 2.7 cm, with a paired 95% interval of
[-0.040, -0.009] m that excludes zero.

**Decisiveness under jitter.** Sweeping restitution with inlet jitter shows
what the deterministic sweep hid. The point design is indecisive at 2 of 20
restitutions, including its own trained value of 0.8, where 12.5% of jittered
draws cross into the wrong bin. The ensemble design is decisive at all 20.

Two honest limits on that. The designs are scored on the same restitutions and
the same jitter draws, so the comparison is paired: two discordant
restitutions against none, and exact McNemar gives **p = 0.50**, the smallest
value two discordant pairs can produce. The direction is consistent and the
failure at the trained value is real, but this comparison has no power to
establish it. And decisiveness means 0 failures in 40 draws per restitution,
which bounds the per-restitution failure rate at about **7.5%**, not at
zero.

