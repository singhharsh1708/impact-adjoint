# Experiments

Seven experiments, each with a committed artifact. Numbers here are mirrored
in [results](results.md), which is generated from those artifacts.

## E3, the failure measured

Three ways to differentiate the same bouncing ball: grid-reset autodiff
(exactly 0.0 at every step size), the interpolated-event repair (converging
but erratic, and itself hand-implemented event sensitivity), and the saltation
endpoint (exact at any step size). See [the problem](problem.md).

## E1, inverse design

Optimize launch velocity and restitution so the ball lands in a cup 1.1 m past
where the initial guess settles, every Adam step one `jax.grad` through both
Tesseracts.

```{image} ../figures/e1_trajectory.png
:alt: Ball trajectories before and after optimization: the initial guess settles short, the optimized launch lands in the cup after five impacts.
:width: 100%
```

```{figure-source} e1_trajectory.png
```

Miss falls from 1.12 m to 2.7 cm through five impacts. The optimizer crosses
bounce-count boundaries on the way (4, 6, 5, with excursions to 8). The
objective is discontinuous at those crossings, which is inherent to contact;
the fixed-topology gradients on each side are exact, and that is what carries
Adam through.

## E2 and E2b, calibration

Recover material parameters `(e, μ)` from the positions of three impacts
observed with 5 mm noise, an observable that exists only because of events.
Point estimation recovers `e` to 0.002 and `μ` to 0.009. For the posterior,
NumPyro NUTS runs against the *containerized* solver: 23,440 leapfrog steps
across two chains, each an apply plus a saltation VJP over HTTP, in about 18
minutes of sampling. Zero divergences, r-hat 1.01,
`e = 0.697 ± 0.007` and `μ = 0.096 ± 0.009`, with the truth inside both
credible intervals.

The container is load-bearing rather than packaging here. NumPyro's jitted
callbacks run off the main thread, which an in-process embedded Julia runtime
does not tolerate; over HTTP the problem does not arise. Reported upstream as
[tesseract-jax#234](https://github.com/pasteurlabs/tesseract-jax/issues/234).

## E4, terrain design

The terrain parameters are differentiable inputs, so the structure itself
becomes the design variable. One terrain routes balls entering at 1.6 and
2.6 m/s to two different cups, miss falling from 2.88 m and 0.48 m to 2.2 and
3.0 cm.

```{image} ../figures/e4_sorter.png
:alt: One fixed terrain with two trajectories: balls entering at 1.6 and 2.6 m/s follow different bounce paths into two separate cups.
:width: 100%
```

```{figure-source} e4_sorter.png
```

## E5, the 24-dimensional separator

Resilience separators sort particles by how far they bounce off a profiled
hard surface. USDA Agriculture Handbook 354 (1968) describes one as "a long
inclined plane interrupted with several bounce plates", and bounce rollers are
sold today to separate potatoes from stones.

Two particles enter identically and differ only in restitution, `e = 0.5` and
`e = 0.8`. The design vector is the 24 bump amplitudes of the surface, and
each particle must land in its own bin.

```{image} ../figures/e5_separator.png
:alt: Left, the designed 24-bump surface with the two material trajectories separating into their own bins; right, the optimizer race in which the gradient-free runs end three to five orders above Adam.
:width: 100%
```

```{figure-source} e5_separator.png
```

Landing errors of 0.23 mm and 0.42 mm. One caveat this shares with any
fixed-length chute: the low-restitution particle is stopped by the event
budget rather than by coming to rest, so the separation surface is "position
after eight impacts", not "position at rest".

For the method comparison, see the multi-seed benchmark in
[verification studies](studies.md), which supersedes the single-run figure
above.

## E5b, design under uncertainty

Real separators process streams with scatter. The design objective becomes the
expected landing loss over an ensemble with inlet-velocity and per-particle
restitution scatter, with fixed common random numbers so the ensemble gradient
is exact. Statistics in [studies](studies.md).

## E6, zero-shot generalization

The E5 surface was optimized for exactly two restitution values. Sweeping the
continuum it never saw, the geometry acts as a classifier: everything in
`e ∈ [0.35, 0.875]` is binned by a single threshold.

```{image} ../figures/e6_generalization.png
:alt: Landing position swept across restitution: a single threshold separates the whole continuum from 0.35 to 0.875, though the surface was trained on only two values.
:width: 100%
```

```{figure-source} e6_generalization.png
```

The binning quantity is where the run ends, which for most in-domain points is
the event-capacity truncation rather than a settled rest position, so the
classifier is "which side after eight impacts". Beyond `e ≈ 0.9`
classification stops being clean, and the jitter study in
[studies](studies.md) shows the point design is less decisive than this
deterministic sweep suggests.
