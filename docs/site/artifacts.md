# Artifacts and provenance

Every number on this site is read out of a file in the repository. This page
is the index of those files: what each one holds, which script writes it, and
which claim it stands behind.

The rule the project holds itself to is that a published figure is never
typed. Where that rule was broken, the [corrections](changelog.md) say so:
a tolerance sweep that was three hardcoded literals, a wall-clock charge
measured over the wrong columns, and a run time on the landing page that read
five minutes against a measured twenty five seconds.

Two of these need Docker: `check_gradients.json`, from Tesseract's own checker
running against the built `contact-sim` image, and `e2b_posterior.npz`, whose
chains run over HTTP against that same container. Everything else regenerates
in process.

```{artifact-index}
```

## Regenerating them

The validation block in [getting started](getting-started.md#validate)
rewrites `oracle_results.json`; `e3_naive_vs_saltation.py` writes `e3_rows.npy`,
and `scripts/time_checks.py` writes `timing.json`. The experiments and studies
rewrite the rest; [run the experiments](getting-started.md#run-the-experiments)
lists them with their run times. `collect_results.py` then rebuilds
`results.json` and `docs/RESULTS.md` from whatever is on disk, and the drift
test fails if the prose no longer matches.

```bash
python experiments/collect_results.py
python -m pytest tests/test_no_drift.py
```
