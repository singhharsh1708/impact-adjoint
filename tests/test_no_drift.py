"""The published numbers must still be the measured ones.

This repository's central claim is that every reported number regenerates from
a committed artifact. That was a convention: `collect_results.py` had to be
remembered after any rerun, and nothing failed if it wasn't. These tests make
it machine-enforced.

They recompute from the artifacts and compare against what is committed. They
never run an experiment, so they are fast enough for CI and cannot themselves
change a result.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
RESULTS_MD = ROOT / "docs" / "RESULTS.md"
RESULTS_JSON = ROOT / "experiments" / "results.json"


def _regenerate(tmp_path):
    """Run the collector against a copy, leaving the working tree alone."""
    before_md = RESULTS_MD.read_text()
    before_json = RESULTS_JSON.read_text()
    try:
        subprocess.run(
            [sys.executable, str(ROOT / "experiments" / "collect_results.py")],
            check=True, capture_output=True, cwd=ROOT,
        )
        after_md = RESULTS_MD.read_text()
        after_json = RESULTS_JSON.read_text()
    finally:
        # the collector writes results.json and RESULTS.md at different points,
        # so a crash between them would otherwise leave one of them rewritten
        RESULTS_MD.write_text(before_md)
        RESULTS_JSON.write_text(before_json)
    return (before_md, after_md), (before_json, after_json)


def test_results_json_matches_artifacts(tmp_path):
    """results.json is what collect_results.py produces from the .npz files."""
    _, (before, after) = _regenerate(tmp_path)
    if before != after:
        b, a = json.loads(before), json.loads(after)
        drifted = {k: (b.get(k), a.get(k)) for k in set(b) | set(a) if b.get(k) != a.get(k)}
        pytest.fail(
            "results.json is stale against the committed artifacts; run "
            f"experiments/collect_results.py. Drifted: {drifted}"
        )


def test_results_md_matches_artifacts(tmp_path):
    """docs/RESULTS.md is regenerated, not hand-edited."""
    (before, after), _ = _regenerate(tmp_path)
    assert before == after, (
        "docs/RESULTS.md is stale against the committed artifacts; run "
        "experiments/collect_results.py"
    )


# Numbers quoted in prose that must keep matching results.json. Each entry is
# (results.json key, formatter, files that quote it).
QUOTED = [
    ("E1_miss_start_m", lambda v: f"{v:.2f}", ["README.md", "docs/writeup.md"]),
    ("CONV_order", lambda v: f"{v:.2f}", ["README.md", "docs/writeup.md"]),
    ("SCALE_r2", lambda v: f"{v:.3f}", ["README.md"]),
    ("SCALE_us_per_param", lambda v: f"{v:.0f}", ["README.md", "docs/writeup.md"]),
    ("SCALE_ratio_77", lambda v: f"{v:.1f}", ["README.md", "docs/writeup.md"]),
    ("BENCH_ratio_eval", lambda v: f"{v:.0f}", ["docs/RESULTS.md"]),
    ("E3_saltation_spread", lambda v: f"{v:.2e}", ["docs/RESULTS.md"]),
    ("GRAD_best_agreement", lambda v: f"{v:.2e}", ["docs/RESULTS.md"]),
]


@pytest.mark.parametrize("key,fmt,files", QUOTED)
def test_prose_quotes_current_value(key, fmt, files):
    """A headline number in prose still matches the artifact it came from."""
    r = json.loads(RESULTS_JSON.read_text())
    assert key in r, f"{key} missing from results.json"
    wanted = fmt(r[key])
    for rel in files:
        text = (ROOT / rel).read_text()
        # a bare substring match lets 1.12 be satisfied by 11.124 elsewhere in
        # the file, and lets a single-digit value match almost any prose
        hit = re.search(rf"(?<![\d.]){re.escape(wanted)}(?![\d])", text)
        assert hit, (
            f"{rel} no longer quotes {key} as {wanted}; either the artifact "
            "moved and the prose was not updated, or the formatting changed"
        )


def test_every_figure_source_path_exists():
    """The figure-provenance mapping cannot point at a renamed artifact.

    The docs build enforces this too (the directive warns, and the build runs
    with -W), so this is a fast local duplicate rather than the only guard. It
    needs Sphinx, which the test job does not install.
    """
    pytest.importorskip("sphinx", reason="enforced by the docs build instead")
    sys.path.insert(0, str(ROOT / "docs" / "site" / "_ext"))
    from figure_source import SOURCES

    missing = []
    for fig, (script, artifacts) in SOURCES.items():
        for rel in [script, *artifacts]:
            if not (ROOT / rel).exists():
                missing.append(f"{fig} -> {rel}")
    assert not missing, f"figure-source mapping points at missing files: {missing}"


def test_sweep_widget_artifact_has_saltation_column():
    """The step-size widget reads a measured saltation column, not a constant."""
    import numpy as np

    rows = np.load(ROOT / "experiments" / "e3_rows.npy")
    assert rows.shape[1] >= 4, (
        "e3_rows.npy has no saltation column; the widget would fall back to "
        "printing a constant. Rerun experiments/e3_naive_vs_saltation.py"
    )
    salt = rows[:, 3]
    assert np.all(np.isfinite(salt))
    # distinct at every step size proves it was solved per dt rather than
    # copied from one solve; a tiny spread is the dt-independence claim itself
    assert len(set(salt.tolist())) == len(salt), (
        "saltation values repeat across step sizes, so the column was filled "
        "from one solve rather than measured at each dt"
    )
    assert float(salt.max() - salt.min()) < 1e-10, (
        "the saltation gradient is no longer dt-independent to the precision "
        "the widget and the studies page claim"
    )


def test_diffrax_claim_is_backed_by_its_artifact():
    """The comparison page's Diffrax numbers come from a committed run."""
    path = ROOT / "scripts" / "diffrax_event_gradient.json"
    assert path.exists(), "diffrax_event_gradient.json missing; run its script"
    d = json.loads(path.read_text())
    assert d["versions"]["diffrax"] and d["versions"]["jax"]
    flat = [g for cfg in d["gradients"].values() for g in cfg.values()]
    assert flat, "no gradients recorded"
    assert any(abs(g - d["expected"]) > 1e-4 for g in flat), (
        "every solver now matches the expected gradient: diffrax may have "
        "fixed this, and docs/site/related.md must be revisited"
    )
