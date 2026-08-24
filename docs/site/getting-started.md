# Getting started

:::{tip}
**Figures in one minute.** Every experiment result is committed as an `.npz`
or `.npy`, so the figures regenerate without rerunning any optimization:

```bash
python experiments/make_figures.py
python experiments/make_e5_figure.py
python experiments/make_e5b_figure.py
python experiments/make_study_figures.py
python experiments/make_design_comparison_figure.py
```
:::

{{ repo_note }}

:::{important}
**This is a reproducible artifact, not a distributed package.** There is no
`pip install impact-adjoint`, and none is planned: the deliverable is a
repository whose every reported number regenerates from committed data.
Cloning is the intended path, and the commands below are the whole of it.
:::

## Requirements

Python 3.12 or newer, Docker for the containerized runs, and about 9 GB of
disk for the images. Julia itself is bootstrapped automatically by
`juliacall`, so there is nothing to install by hand.

```bash
git clone https://github.com/singhharsh1708/impact-adjoint
cd impact-adjoint
python3 -m venv .venv && source .venv/bin/activate
pip install -r docs/requirements-repro.txt
```

:::{note}
The first Julia call bootstraps a project environment, and Julia itself if it
is absent. Expect one to five minutes and a wall of `[juliapkg]` output the
first time. Warm runs take seconds.
:::

## Validate

No Docker needed for any of this.

```bash
python scripts/proof_local.py         # 5 s boundary proof
python scripts/validate_contact.py    # FD gate, robustness
python scripts/validate_closed_form.py  # symbolic oracle
python scripts/validate_reference.py  # independent scipy impl
pytest tests/                          # 45; 4 skip without Sphinx
```

## Run the experiments

```bash
python experiments/e3_naive_vs_saltation.py
python experiments/e1_inverse_design.py
python experiments/e2_calibration.py
python experiments/e4_terrain_design.py
python experiments/e5_separator.py            # about 30 s warm
python experiments/e5b_robust_separator.py    # about 10 min
python experiments/e6_generalization.py
```

## Containerized

```bash
tesseract build tesseracts/contact_sim
tesseract build tesseracts/score_target
tesseract run contact-sim check-gradients \
    @tesseracts/contact_sim/check_payload.json
python scripts/capture_check_gradients.py   # same check, seeded, writes the artifact
python experiments/e2b_bayesian.py                   # NUTS, 2 chains
python experiments/e1_inverse_design.py --container  # served
tesseract serve -p 8123 contact-sim &                # curl client needs a server
./scripts/second_client_curl.sh                      # curl only
```

## Troubleshooting

```{list-table}
:header-rows: 1

* - Symptom
  - Cause and fix
* - `tesseract` runs an OCR tool
  - PATH collision with Tesseract-OCR. Use the venv's `tesseract` binary.
* - Wall of `[juliapkg] Installing...`
  - One-time Julia bootstrap. Later runs are warm.
* - `docker build` fails with `unknown flag: --load`
  - Docker buildx plugin missing, common with colima and podman setups.
* - Container cannot write outputs
  - Pass an `output_path` under your home directory. VM file sharing may not
    cover system temp directories.
```
