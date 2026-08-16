# Changelog

Notable changes to the solver, the experiments, and any number this project
reports. Corrections are listed as prominently as features: where a published
figure changed, this says what it was and what it became.

## Unreleased

### Corrected

- **E5 and E4 reported a design they had not scored.** Three optimization
  scripts returned the parameter vector one update past the last point they
  evaluated, so the objective in the trace belonged to a different design than
  the one saved, plotted and quoted. They now return the best evaluated
  iterate. The E5 landing errors move from 0.38 mm and 0.34 mm, which were
  measured on the unevaluated design, to **0.23 mm and 0.42 mm**. The
  optimizer benchmark was already correct, so the 917x and 24x figures are
  unaffected.
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
