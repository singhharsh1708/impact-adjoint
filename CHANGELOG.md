# Changelog

Notable changes to the solver, the experiments, and any number this project
reports. Corrections are listed as prominently as features: where a published
figure changed, this says what it was and what it became.

## Unreleased

### Corrected

- **The tolerance sweep in `check_gradients.json` was three hardcoded literals.**
  They were written into the artifact whose entire purpose is that its numbers
  are measured, and the README quoted them. The script runs the sweep now, and
  running it changed the answer: at the same `eps = 1e-6` as the headline run,
  the checker first fails at `rtol = 1e-5` with 22 of 50, not at `1e-6` with 13.
  The earlier figure came from an ad-hoc sweep at a different `eps`.
- **Four upstream fixes have merged since the last entry.** tesseract-core #667
  and tesseract-jax #236 are both merged, and issues #666 and #235 are closed;
  only the juliacall deadlock #234 is still open. The pages said all the
  Tesseract work was still open.

- **A README claim about our own gradient checker was false.** It said the
  check fails at `rtol = 1e-5`. It passes there; the first failure is at
  `1e-6`, 13 of 50, which is where the central difference stops resolving
  rather than where the gradient does. The tolerance sweep is recorded in
  `check_gradients.json` now instead of asserted.
- **The results table published the statistic the prose calls wrong.** The
  seeds are paired, so the ratio of medians (917x) pairs one seed's CMA with
  another seed's Adam; every prose page already used the paired per-seed
  median (347x) while the generated table led with the unpaired one. The table
  now leads with the paired statistic and keeps the unpaired one labelled as
  reference.
- **"1.1 to 5.1 orders" was CMA-ES alone, attributed to both gradient-free
  methods.** Nelder-Mead spans 2.9 to 8.0. The published range is 1.1 to 8.0
  and the collector computes it over both methods.
- **E3_truth was a hardcoded literal** in the file whose premise is that
  nothing is retyped, so a uniform solver shift would have left it and both
  derived errors untouched. It is the mean of the measured per-dt saltation
  column now.
- The `e3_bias` figure drew the saltation reference as a literal `1e-15` line,
  669x below what the artifact measures. It plots the measured column.
- The robustness figure's caption still said the ensemble objective buys
  margin "not headline purity"; after the rebuild purity improves with
  McNemar p = 0.0026, so the caption said the opposite of the panel beneath it.
- E2b sampling time read "18 minutes" in four places against a measured 22.7,
  and the test count read 21 against 33 collected.

- **The NUTS convergence statistic was the outdated one, and the number a
  reader actually needs was missing.** `gelman_rubin` is the classic
  non-split R-hat, which cannot detect within-chain non-stationarity; it is
  `split_gelman_rubin` now. Effective sample size was never reported at all:
  2000 draws buy **344 and 330** effective ones, which is the relevant figure
  given the sampler cost is a headline. Two chains estimate the between-chain
  variance on one degree of freedom, so R-hat here is stated as a weak check
  rather than a passed one. The wall-clock figure timed warmup plus sampling
  while the leapfrog count covered sampling only; the label says so now.

- **The gradient-free designs had never been scored on the metric this project
  says matters, and doing so weakens a headline.** E5 saved the CMA-ES and
  Nelder-Mead designs and nothing ever loaded them. Scored on E5b's held-out
  scatter ensemble: Adam 0.995 purity with a 0.07 m fifth-percentile margin,
  CMA-ES 0.980 with 0.18 m, and **Nelder-Mead 0.995 with 0.41 m** — five orders
  behind on the objective, identical purity, wider margin. In engineering units
  the four-order loss gap is 0.34 mm against 12.6 mm with bins 1600 mm apart.
  "The gradient-free methods never reach the design" is now stated as what it
  is, a statement about objective value, with the translation shown.
- **The ensemble design was built from a superseded point design.** E5b starts
  from E5's output, and its committed result predated the fix that made E5
  return the iterate it had actually scored. Rebuilt: held-out purity is
  **997/1000**, not 1000/1000, and the fifth-percentile margin improves
  0.05 m to 0.49 m rather than to 0.58 m.
- **The worst-case margin claim was backwards.** It said the worst case moves
  "from inside the wrong bin to 0.09 m clear". On the rebuilt designs it goes
  -0.12 m to -0.35 m, both inside the wrong bin. The ensemble
  objective buys the low tail, not the extreme, and the page says so now.
- **Two statistical methods were wrong even where the answer held.** The
  bootstrap resampled the two margin arrays independently although the same
  1000 particles are scored under both designs; it now resamples particle
  indices and carries both together. Purity is a paired binary outcome, so
  non-overlapping Wilson intervals were the wrong criterion; McNemar is
  reported alongside, and gives p = 0.0026.

- **The novelty claim was too broad and two thirds of it is withdrawn.** A
  prior-art sweep found that saltation matrices have been used for optimal
  *design* explicitly (Kong et al., Proc. IEEE 2024), that a pip-installable
  package already pairs them with adjoint-gradient trajectory optimization,
  that SciMLSensitivity.jl implements exact event-time corrections with a
  bouncing-ball restitution-optimization example, that Diffrax gained
  pathwise event-time gradients in 2024, and that FMI has carried derivatives
  across a component boundary since 2014, with adjoint derivatives added for
  machine-learning consumers in 2022. Tesseract's own fortran_enzyme example
  is the same boundary pattern. What is claimed now is only the conjunction:
  gradient-based design of passive geometry, with exact event times, against
  a per-object routing objective, with the nearest neighbours named.
- **The gradient-checker headline measured a sampler, not a gradient.** The
  CLI defaults to `rtol = 0.1` and samples with replacement, so "0 failures /
  1574 checks" was mostly repeated comparisons at ten percent tolerance. It
  now runs distinct entries at `rtol = 1e-4`: **0 failures / 50 checks** on
  each of the three endpoints. It fails at `1e-5`, where the finite difference
  itself stops resolving, so 1e-4 is the tightest honest setting.

- **We were wrong about Diffrax, in its favour.** This project claimed the
  restart-after-event pattern returned solver-dependent wrong gradients,
  citing three numbers from Diffrax issue #729. Reading the whole thread, the
  maintainer diagnosed the reproducer as a usage error (the jump time was
  passed to `ClipStepSizeController` as a plain float rather than closed over
  differentiably) and noted the vector field was not valid input; the reporter
  later said the example may not reproduce what they were chasing. Our own
  script repeated the same mistake. Measured correctly, **Diffrax returns the
  exact gradient under all three solvers**. The comparison now says so, and
  the remaining difference is that Diffrax has no reset map, which is
  ergonomic rather than a correctness gap.
- **The wall-clock optimizer comparison over-charged the gradient.** The
  charge of 8.5 forward solves came from a VJP over every input, 77
  sensitivity columns, but the separator differentiates only its 24 bump
  amplitudes. The benchmark now measures the charge against the inputs it
  actually differentiates: 6.8 solves. CMA-ES is **2.3x** ahead on the ratio
  of medians under
  wall-clock accounting, not the 24x previously published.
- **E6 and the generalization study evaluated the design past its own
  horizon.** They redeclared `t_final = 3.0` where E5 designs at 2.2, so the
  separator was scored two impacts beyond the point its output is defined at.
  At its own trained value e = 0.8 the design lands at 4.3996 against a 4.4
  target; evaluated at the longer horizon it read 3.7262, which is the number
  that had been published. Both now import the design's configuration instead of
  restating it. The point design is indecisive under jitter at **2 of 20**
  restitutions, not 5 of 20, still including its own trained value; at
  e = 0.85 it is now unanimous where two different samples had disagreed.
- **"Reported and fixed upstream" was not true.** Two Mosaic harness fixes are
  merged; the three Tesseract issues and the two PRs against them are open.
- The writeup said the solver "raises an explicit error at degenerate
  crossings". It does not: the guard on the saltation denominator is
  effectively unreachable, and the real behaviour is a large finite gradient
  and then a silently missed event. Stated as a limitation now.
- Zhong et al. (NeurIPS 2021) was filed under "learned contact models". Its
  contact model is analytic; only its parameters are learned.
- The Hiskens & Pai parameter augmentation is equations 13 to 16 with the
  augmented sensitivity system at 33, not 62 to 63.
- `drag` and `v_stop` were unvalidated while `e > 1` was rejected for exactly
  the same reason. Negative drag added energy without bound and was accepted.

- **E5 and E4 reported a design they had not scored.** Three optimization
  scripts returned the parameter vector one update past the last point they
  evaluated, so the objective in the trace belonged to a different design than
  the one saved, plotted and quoted. They now return the best evaluated
  iterate. The E5 landing errors move from 0.38 mm and 0.34 mm, which were
  measured on the unevaluated design, to **0.23 mm and 0.42 mm**. The
  optimizer benchmark's own loop was already correct, so the 917x eval-accounting
  figure is unaffected. (The wall-clock figure changed later, for the separate
  reason recorded above.)
- **The performance table did not match its artifact.** It listed 4.9 ms and
  41.6 ms for 77 parameters where `scaling_result.npz` records 2.34 ms and
  19.82 ms, and labelled the column with the wrong configuration. The table is
  now generated from the artifact.
- **E2b's diagnostics rested on nothing.** The divergence count, r-hat and
  throughput were prose. The run now records them: 23,440 leapfrog steps over
  two chains in 23 minutes of warmup plus sampling, 0 divergences, non-split r-hat 1.0089 and 1.0119 (now reported as split r-hat, 1.0061 and 1.0074). The
  previous estimate of "roughly 10,000 solver calls per chain" is replaced by
  the measurement.
- Held-out purity now states which figure comes from one 200-particle draw
  (199/200) and which from five independent ensembles (983/1000), which read
  as contradictory before.
- In-figure series labels were below the WCAG AA contrast threshold for text
  (4.42:1 and 3.20:1 on white). Label colours are darkened; plotted line
  colours are unchanged, so the figures are otherwise identical.

### Added

- Documentation site at <https://impact-adjoint.vercel.app>, with a component
  reference generated from the Tesseract schemas, an interactive step-size
  sweep driven by the committed E3 artifact, and a comparison against Diffrax,
  MuJoCo MJX, Brax, DiffTaichi, NVIDIA Warp and Dojo.
- Every figure now names the artifact it was plotted from and the script that
  drew it.

## 0.1.0

First complete entry: three Tesseracts, seven experiments, six verification
studies, twelve golden tests, three independent oracles, and CI on every push.
