# Changelog

Notable changes to the solver, the experiments, and any number this project
reports. Corrections are listed as prominently as features: where a published
figure changed, this says what it was and what it became.

## 0.1.3 (2026-08-25)

### Note on 0.1.0 and 0.1.1

0.1.1 shipped a crashing command and three stale figures, both fixed here; it
is superseded but its tag is left where it is.

### Note on 0.1.0

0.1.0 was tagged earlier the same day and is superseded. Its E5b design was
scored, but it was not the best available: the correction it shipped discarded
Adam's final update, which the loop never scored and which is better, so 0.1.0
published a worse design than the one it replaced. Its three E5b figures were
also never regenerated after that correction, so they drew the numbers of a
still earlier design, and its gradient-checker sweep was sampled without a
seed. All of that is corrected here. The tag stays where it is rather than
being moved, so the record of what it published survives.

### Corrected

Every entry below is a correction made while building 0.1.0 or 0.1.1, written
as "it was X, it is now Y", newest first. They are a record of what moved, not
a statement of the current values: several entries quote figures that later
entries supersede, and one correction is itself withdrawn further up.
For what the entry publishes today, read `docs/RESULTS.md`, which is generated
from the committed artifacts, or the artifacts themselves.

- **Five published statistics had no code computing them.** The four per-design
  rows on the studies page carried Wilson and bootstrap intervals that existed
  only as typed text, and the jitter-sweep McNemar p was quoted in two places
  with nothing producing it. `study_design_table.py` generates the table into
  an artifact the suite compares the page against, and the McNemar is computed
  where the sweep is. One published bound moved: Nelder-Mead's upper interval
  is 0.52, not 0.51.
- **The repeat-run figures were quoted from runs that were never recorded.**
  A slope "re-measured at 81 µs", a gradient charge of "5.5 forward solves" and
  a "ratio of medians of 1.3x" appeared across three pages with no artifact
  behind any of them. The scaling study is run twice now and both runs are
  committed: 93.1 µs and 83.0 µs, an 11% spread. The two figures that had no
  second run are gone rather than restated.
- `make_e5_figure.py` restated the entire E5 configuration instead of importing
  it, and its copy had silently dropped `v_stop`, so the published figure
  re-solved under the schema default rather than the value the design declares.
- Two dead literal fallbacks, `... else 0.09037774` in the collector and the
  figure script, were one three-column rewrite away from republishing a stale
  truth value through every page that quotes it. They fail loudly now.
- The entry has eight labelled experiments and seven verification studies, not
  seven and six.

- **A published README command crashed.** `e6_generalization.py` used
  `BIN_RUBBER` and `BIN_PET` seven lines above the import that defines them,
  so it died with a `NameError` before doing anything. It is the seventh
  command a reader following the README types, and it shipped because CI ran
  the tests, the validators and the fuzzer but never an experiment. The four
  fast experiments run in CI now.
- **Three figures published the numbers of a withdrawn design.** The artifacts
  were restored and the prose rewritten, but `study_robustness.png`,
  `design_comparison.png` and `e5b_purity.png` were never regenerated, so the
  first read "983/1000 vs 1000/1000" against an artifact saying 997, and the
  second drew the ensemble design at 0.37 m below Nelder-Mead while the
  sentence above it said 0.43 m and "the widest tail of the four". Two guards
  checked that a figure names an artifact and that the artifact exists; neither
  checked the picture was current. A test regenerates the artifact-only figures
  and compares bytes now.
- **The 0.1.0 note had the defect backwards, in three places.** The changelog,
  the annotated tag and the release body all said 0.1.0 shipped "a design the
  optimizer never scored". It did not: 0.1.0's design was scored, and was the
  best point inside the loop. The unscored one was Adam's final update, which
  0.1.0 discarded and which is better. The design shipping now is bit-identical
  to the one predating both corrections.
- **The README's justification for the gradient-checker settings was false.**
  It claimed the run uses "distinct entries" against the CLI default's repeats.
  The checker samples with replacement whatever the budget, and the payload has
  14 differentiable elements over two output paths, so 150 checks are about 76
  distinct comparisons. Seeding fixed repeatability, not representativeness:
  other seeds give 12, 20 and 21 failures at `rtol = 1e-5` where this one gives
  14. Both are stated now, and the seed is disclosed.
- **The capture script could destroy the artifact it exists to produce.** It
  wrote the file and then asserted the run was clean, so a failing run replaced
  the committed artifact with its own failures before raising. It validates
  first now, checks every endpoint reported, sums the sweep denominator from
  the rows it parsed rather than multiplying one of them, and fails loudly
  instead of silently dropping a tolerance that produced no output.
- **Whether a degenerate crossing raises was documented two contradictory
  ways**, and both were wrong. Tangency cannot reach the guard: the denominator
  floors near `1e-7` on metre-scale terrain, five orders above the `1e-12`
  threshold, and chatter cannot either, because the re-arm lift floors the
  approach speed at `1.4e-6`. The guard is reachable, through a launch height
  small enough that the first impact speed falls under `1e-12`. The second
  guard, on non-approaching velocity, fires only past the RK4 stability limit
  and is unreachable through the schema. All of that is written down now.
- The drift guard lost coverage in its rewrite: the checker's denominator could
  be republished as the old single-endpoint 50 with the suite green, and the
  "14 of 150" headline was guarded nowhere. Both are pinned now, as is the
  stability limit quoted in the schema descriptions.
- Light-mode contrast: the sweep widget's two value colours sat at 2.55:1 and
  2.90:1, and the footer provenance, sidebar captions and figure-source lines
  at 3.26:1 and 3.50:1, all under the 4.5:1 threshold for text. The text tones
  are darkened; the figure hues are unchanged.
- The artifacts page named "three independent oracles" where the write-up says
  two solver-independent ones plus a finite-difference gate, and rendered that
  on the live site.
- Four tests skip without Sphinx, not three; the run time moved 24.8 s to
  31.6 s on a busier machine; and `make_study_figures.py`'s title and
  `study_generalization_stats.py`'s docstring were both left describing the
  withdrawn design.

- **The E5b iterate fix was itself wrong, and is withdrawn.** Keeping the best
  point scored inside the loop discarded Adam's final update, which the loop
  never scored and which is 0.4% better on the objective. So the correction
  published a worse design and moved eight numbers for nothing: the five below
  plus the five-ensemble McNemar p (1.5e-05 back to 0.0026), the jitter-sweep
  McNemar p (1.0 back to 0.50), and the median margin (+1.8 cm back to
  -2.7 cm). The design that ships now is bit-identical to the one that
  predates both corrections; what changed is that it is scored. The loop now
  scores the final iterate too and keeps the best of everything evaluated,
  which restores the original design with real provenance: purity is
  **997/1000**, the worst case **-0.35 m**, the fifth percentile **0.49 m**,
  the ensemble margin **0.43 m**, and the jitter sweep **20 of 20**. Every
  page that was changed for the bad fix is changed back.
- **The gradient checker was sampling unseeded.** The same sweep re-measured
  15, 17, 19 and 21 failures at `rtol = 1e-5` across four runs, so a published
  count was one draw. The seed is pinned and recorded in the artifact now, and
  two runs at the same seed agree exactly: **14 of 150**.
- **The headline `checks` and `failures` still mixed populations.** The sweep
  was fixed but the headline pair was not: `checks` counted one endpoint while
  `failures` summed three, hidden only because the count is zero. Both are
  totals now, and the pages say "across the three gradient endpoints".
- **The prose guard accepted 11 of 19 wrong values.** Anchoring a bare value
  to a context string within a two-line window was barely stronger than
  matching it anywhere: the point-design purity guard was satisfied by the
  ensemble design's number in the same sentence, and the 93-microsecond guard
  by the 81 published two lines below it. Each entry now carries the phrase the
  number belongs to, and the value has to appear inside it.
- **A superseded purity survived in `docs/writeup.md` and the guard could not
  see it.** The guard proves the current value appears somewhere; it cannot
  detect a stale duplicate elsewhere in the same file. A second check now
  requires every occurrence of a quantity written in a fixed shape to be
  current.
- **The drift test deleted two committed files in place.** SIGTERM, SIGHUP or
  SIGKILL inside a 0.13 second window left the working tree with both gone,
  a failed first restore skipped the second, and it broke under parallel
  execution. It runs against a copy now and never writes the real tree.
- **The landing-page timing test passed for "about 25 minutes" against 24.8
  seconds**, because it compared digits and not the unit, and its stat-card
  half only checked that a key name appeared in a table. Both halves render
  and compare now, and it reads the page it is named for.
- **Two tests failed under the install the README documents.** They import
  `docutils`, which `requirements-repro.txt` does not carry; CI installs
  Sphinx separately and hid it. They skip now, and the counts say three tests
  need Sphinx rather than one.
- **The RK4 stability bound was a rounded guess, untested, and misordered.**
  2.7 rejected a genuinely stable band up to 2.785293563, the real root of
  `z^3 + 4z^2 + 12z + 24`; replacing the constant with 999 or with 0.001 left
  the suite green; and an infinite drag reported a stability violation rather
  than a finiteness one. The limit is computed rather than typed, straddled by
  a boundary test, checked after the finiteness loop, and stated in the schema
  descriptions the reference page is generated from.
- **Berkowitz and Canny's objective is not per-object rather than bulk.** They
  score a per-trajectory metric alongside a feed-rate efficiency taken over
  all initial orientations. Both the write-up and the related-work page said
  otherwise, so making them agree had propagated the error rather than fixing
  it.
- The write-up's snippet claim that "the only difference is where the impact is
  applied" stopped being true once the second snippet gained an x64 line, which
  it needs because the component returns Float64.
- `figure_source` indexed 10 of the 12 committed figures; the two it missed are
  both displayed in the write-up. A test now fails if a figure has no source.
- The README's figure tip named three scripts for what takes five, and called
  twelve figures eight.
- The median-margin interval was published with no code computing it. The
  robustness study bootstraps it now, and `collect_results.py` collects the
  McNemar p, the tail interval and the median interval it was already storing
  but never reading.
- `experiments/timing.json` predated its own generator, still carrying the
  literal `depot` field the fix replaced with a derived one.
- The `citing` page promised a DOI "as soon as v0.1.0 is tagged", after it was.
- A dangling sentence fragment in this file, orphaned when the lines above it
  were rewritten, pointed its "superseded by" at the wrong entry.

- **The ensemble design was one Adam step past the last point it evaluated.**
  E5 and E4 were fixed for this; `e5b_robust_separator.py` was not. Its
  training ensemble is drawn once from a fixed seed, so the objective is
  deterministic in the design and the best evaluated iterate is the right one
  to keep. Correcting it moves published numbers in both directions. Held-out
  purity over five ensembles goes **997/1000 to 1000/1000** and the worst case
  **-0.35 m to +0.07 m**, out of the wrong bin rather than inside it; the
  fifth-percentile margin goes 0.49 m to **0.40 m** and E5b's ensemble margin
  0.43 m to **0.37 m**, which is now narrower than Nelder-Mead's 0.41 m. The
  median margin no longer costs anything: it improves 1.8 cm, where the
  previous design lost 2.7 cm.
- **The jitter sweep no longer separates the two designs, and the pages said it
  did.** The corrected ensemble design is decisive at 18 of 20 restitutions,
  not 20 of 20. The point design is also 18 of 20. Two discordant pairs against
  two gives exact McNemar **p = 1.0**, where the pages reported p = 0.50 and a
  consistent direction. Robustness bought on the scatter ensemble did not
  transfer to restitutions far from the training distribution, and that is
  what the pages say now.
- **The tolerance sweep counted failures over three endpoints against one
  endpoint's checks.** `check_gradients.json` recorded 81 failures out of 50
  checks at `rtol = 1e-7`, which is impossible and was the tell. The README
  published "22 of 50" for a rate whose denominator is 150. The sweep records
  the endpoint count and both denominators now, and re-measuring gives **15 of
  150** at `rtol = 1e-5`.
- **The docs CI gate was not a gate.** `docs.yaml` piped Sphinx through `tee`
  with no `shell:` key, so Actions ran it under `bash -e` without `pipefail`
  and the step took `tee`'s exit status. `-W` never failed a build, which
  silently disarmed every provenance guard in `docs/site/_ext`, all of which
  are `logger.warning` and only fatal under `-W`. The drift test explicitly
  delegates to this gate.
- **Two drift tests passed when the collector did nothing.** They compared
  `RESULTS.md` and `results.json` before against after, so stubbing
  `collect_results.main()` with a bare `return` made all twenty pass. Both
  files are deleted before the run now.
- **Seven of thirteen prose guards were checking a generated file against its
  own generator**, and the search took the first match anywhere in the file, so
  a nearby number satisfied it: README quotes 81 as a repeat measurement two
  lines under the 93 the test guards. Each entry now names the sentence the
  number belongs to. Setting `point_k` to 900 previously left five pages
  publishing 983 with the suite green; it fails now.
- **`drag` and `dt` were validated independently, never their product.** The
  integrator is fixed-step explicit RK4, so it is stable only while
  `drag * dt` stays under about 2.785. Past that a purely dissipative force
  gains energy, sensitivities change sign, and the run still reports status 0.
  Rejected at validation now.
- **`abstract_eval` accepted a negative `n_samples`** and returned a shape of
  `(-3, 5)`, disagreeing with `apply` on the same input. `n_samples` is a plain
  int and stays concrete on the abstract path, so it is checked before the
  early return.
- **The writeup contradicted the repository's own related-work page on two
  cited works.** DiffTaichi was filed under engines that trade away
  event-gradient fidelity, when it documents this exact failure and repairs it
  with a precise time of impact, and is the closest prior art here. Roussel et
  al. and Berkowitz & Canny were described as differentiating a smoothed model
  against a bulk flow statistic; both are non-gradient, and Berkowitz & Canny's
  objective is per-object. The withdrawn "our contribution is the packaging"
  claim was also still standing.
- **"tuned CMA-ES" described an untuned design.** The CMA-ES design E5b scores
  comes from a single hardcoded `sigma0 = 0.05, seed = 3` in `e5_separator.py`,
  not from the tuning grid. Only `study_optimizers.py` tunes.
- `collect_results.py` wrote the E2b draw count as a literal `2000` disguised
  as `r.get(...) and 2000`, in the table whose header says nothing is retyped.
  `n_draws` was in the artifact all along.
- The E2b wall time was labelled "sampling" in the writeup and the generated
  table; it covers warmup plus sampling. The leapfrog count is the sampling
  phase alone, and `experiments.md` had paired the two.
- `time_checks.py` wrote `"depot": "warm"` unconditionally, in a file whose
  docstring says the depot state is recorded rather than assumed. It is derived
  from whether the first repeat is anomalously slow.
- The landing page's closed-form card floored the exponent, rendering 7.1e-12
  as 10^-12 and claiming a tighter agreement than the artifact supports. It
  shows the measured value, like the oracle card beside it.
- Four regression tests were weak: a one-hot cotangent made the VJP check
  algebraically identical to reading the first Jacobian row, two finite
  difference comparisons sat inside an `if` that skipped them silently on a
  topology shift, and the bounds test used a bare `Exception` over four of ten
  validator branches. All eleven branches are exercised against their messages
  now.
- The landing-page timing test never read the landing page. Replacing the
  substitution in `conf.py` with a literal passed it.
- `study_generalization_stats.py` printed a self-contradicting verdict
  ("unanimous at 18/20") once the design stopped being unanimous.
- The `artifacts.md` page said only `check_gradients.json` needs Docker;
  `e2b_posterior.npz` does too. Its regeneration note also credited the wrong
  commands for `e3_rows.npy` and `timing.json`.
- The second landing-page snippet is documented as running as written, but
  needs `jax.config.update("jax_enable_x64", True)`; the component returns
  Float64 and JAX refuses without it.
- Test counts read 33 against 36 collected, in the README and getting started.

- **The wall-clock optimizer comparison rests on one measured scalar, and that
  scalar moves.** The gradient charge re-measured at 5.5 forward solves rather
  than 6.8 on a repeat run, which moves the ratio of medians from 2.3x to 1.3x.
  Every optimizer trace in the artifact is bit-identical between runs; only the
  timing moved. The pages already reported the wall-clock ordering as
  unresolved at n = 5, and they now say the charge itself is a measurement with
  spread.
- `scripts/second_client_curl.sh` failed with `Expecting value: line 1 column 1`
  when no server was running, which describes nothing. It checks `/health`
  first and names the command to start one.

- **The landing page said the four checks take about five minutes; they take
  about twenty five seconds.** The five minutes was the first-run Julia
  bootstrap, not the checks, and the figure came from no artifact. It is
  measured now by `scripts/time_checks.py`, which writes `timing.json`, and
  both the stat card and the prose read it.
- **The landing page's headline figures were hand-typed HTML.** The drift test
  covers README.md, docs/writeup.md and docs/RESULTS.md, so the five numbers on
  the most-read page of the site were the one place a published figure could go
  stale silently. They are generated from `results.json` now. The other four
  were correct.
- **The cost-scaling slope was quoted as though it were precise.** It is
  wall-clock on one shared laptop: re-running the study moved 93 microseconds
  per parameter to 81, and R² from 0.998 to 0.997. The affine shape holds; the
  absolute figures carry ten to twenty percent of run to run spread, and the
  pages that quote them say so.
- `study_generalization_stats.py` printed `transition band ... e in [nan, nan]`
  when the robust design classified every sampled restitution unanimously.
  That is the good case, and it now says so in words.

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
  check fails at `rtol = 1e-5`, and this entry claimed it passes there.
  **Both were wrong**: that sweep ran at a different `eps` than the headline
  configuration. At the published `eps = 1e-6` the checker does fail at
  `1e-5`. Superseded by the sweep-denominator entry at the top of this
  section, which re-measures it as 15 of 150 across the three endpoints.
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
  CMA-ES 0.980 with 0.18 m, and **Nelder-Mead 0.995 with 0.41 m**: five orders
  behind on the objective, identical purity, wider margin. In engineering units
  the 3.9-order loss gap is 0.34 mm against 31 mm with bins 1600 mm
  apart. (This entry first quoted 12.6 mm, which came from the multi-seed
  benchmark median rather than the CMA-ES design E5b actually scores.)
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

- An [artifacts and provenance page](https://impact-adjoint.vercel.app/artifacts.html)
  indexing every committed artifact against the script that writes it and the
  claim it backs, with a test that fails if an artifact is added without a row
  or a row points at a file that is gone.

- Documentation site at <https://impact-adjoint.vercel.app>, with a component
  reference generated from the Tesseract schemas, an interactive step-size
  sweep driven by the committed E3 artifact, and a comparison against Diffrax,
  MuJoCo MJX, Brax, DiffTaichi, NVIDIA Warp and Dojo.
- Every figure now names the artifact it was plotted from and the script that
  drew it.

### Baseline

First complete entry: three Tesseracts, eight experiments, seven verification
studies, twelve golden tests, two solver-independent oracles plus a
finite-difference gate, and CI on every push.
The suite has grown since; the README states its current size.
