# Changelog

Notable changes to the solver, the experiments, and any number this project
reports. Corrections are listed as prominently as features: where a published
figure changed, this says what it was and what it became.

## Unreleased

### Corrected

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
  two chains in 18 minutes, 0 divergences, r-hat 1.0089 and 1.0119. The
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
