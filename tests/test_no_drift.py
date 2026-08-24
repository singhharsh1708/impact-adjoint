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
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
RESULTS_MD = ROOT / "docs" / "RESULTS.md"
RESULTS_JSON = ROOT / "experiments" / "results.json"


def _regenerate(tmp_path):
    """Run the collector against a copy of the repo and return what it wrote.

    An earlier version deleted the two committed files in place and restored
    them in a `finally`. That covered exceptions but not signals: SIGTERM,
    SIGHUP or SIGKILL inside the window left the working tree with both files
    deleted, and a failed first restore skipped the second. It also made the
    file unsafe to run in parallel with its own siblings, which read those
    files. Copying first means the real tree is never written at all.
    """
    work = tmp_path / "repo"
    work.mkdir(parents=True, exist_ok=True)
    for rel in ("experiments", "docs"):
        shutil.copytree(ROOT / rel, work / rel, dirs_exist_ok=True)

    # Copying brings the previous outputs along, so their presence proves
    # nothing. Delete them inside the copy: a collector that writes nothing
    # then leaves a missing file rather than passing on its own stale output.
    (work / "docs" / "RESULTS.md").unlink()
    (work / "experiments" / "results.json").unlink()

    done = subprocess.run(
        [sys.executable, str(work / "experiments" / "collect_results.py")],
        capture_output=True, text=True, cwd=work,
    )
    assert done.returncode == 0, (
        f"collect_results.py failed in a clean copy:\n{done.stdout[-2000:]}\n"
        f"{done.stderr[-2000:]}"
    )
    out_md = work / "docs" / "RESULTS.md"
    out_json = work / "experiments" / "results.json"
    assert out_md.exists(), "collect_results.py did not write docs/RESULTS.md"
    assert out_json.exists(), "collect_results.py did not write results.json"

    return ((RESULTS_MD.read_text(), out_md.read_text()),
            (RESULTS_JSON.read_text(), out_json.read_text()))


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


def _sci(v, digits):
    """Scientific notation the way the prose writes it, without a padded exponent."""
    mantissa, exponent = f"{v:.{digits}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# Numbers quoted in prose that must keep matching results.json. Each entry is
# (key, formatter, [(file, phrase template)]), where the template renders with
# the current value and must appear verbatim.
#
# Two weaker designs came before this. Matching the bare value anywhere in a
# file let an unrelated number satisfy the guard. Anchoring to a context
# string within a two-line window was barely better: a sweep over candidate
# artifact values found that 11 of 19 entries still accepted at least one
# wrong value, because every sentence here carries several numbers. The
# point-design purity guard was satisfied by the ensemble design's number in
# the same sentence, and the 93-microsecond guard by the 81 published two
# lines below it. Requiring the whole phrase removes the ambiguity: the value
# has to appear inside the claim that is making it.
QUOTED = [
    ("E1_miss_start_m", lambda v: f"{v:.2f}", [
        ("README.md", "Miss **{v} m"),
        ("docs/writeup.md", "Miss distance falls **{v} m"),
        ("docs/site/experiments.md", "Miss falls from {v} m"),
    ]),
    ("CONV_order", lambda v: f"{v:.2f}", [
        ("README.md", "Order {v} on a smooth arc"),
        ("docs/writeup.md", "converges at **order {v}**"),
        ("docs/site/studies.md", "Order {v} on a smooth arc"),
    ]),
    ("SCALE_r2", lambda v: f"{v:.3f}", [
        ("README.md", "at R\u00b2 = {v}"),
        ("docs/writeup.md", "(R\u00b2 = {v}"),
        ("docs/site/studies.md", "R\u00b2 = {v}"),
        ("docs/site/method.md", "R\u00b2 = {v},"),
    ]),
    ("SCALE_us_per_param", lambda v: f"{v:.0f}", [
        ("README.md", "{v} \u00b5s per parameter"),
        ("docs/writeup.md", "{v} microseconds per"),
        ("docs/site/studies.md", "{v}\nmicroseconds per parameter"),
        ("docs/site/method.md", "({v} microseconds per parameter"),
    ]),
    ("SCALE_ratio_77", lambda v: f"{v:.1f}", [
        ("docs/writeup.md", "{v}\u00d7 for"),
        ("docs/site/method.md", "{v}x for 77"),
    ]),
    ("CHECKGRAD_failures", lambda v: f"{int(v)} failures", [
        ("README.md", "**{v} /"),
    ]),
    # The denominator lost its guard in the rewrite: README could publish the
    # old single-endpoint "50 checks" with the suite green.
    ("CHECKGRAD_checks", lambda v: str(int(v)), [
        ("README.md", "/ {v} checks** across the three gradient endpoints"),
    ]),
    # The seeded sweep's headline, previously guarded nowhere.
    ("CHECKGRAD_first_failing_count", str, [
        ("README.md", "({v} of 150 checks across the three endpoints)"),
    ]),
    ("CLOSED_FORM_jacobian_worst", lambda v: _sci(v, 0), [
        ("README.md", "Jacobian agreement **{v}**"),
    ]),
    ("REFERENCE_jacobian_worst", lambda v: _sci(v, 0), [
        ("README.md", "**{v}**, covering"),
    ]),
    ("CONTACT_energy_drift", lambda v: _sci(v, 0), [
        ("README.md", "drift {v})"),
    ]),
    ("ROBUST_point_correct", lambda v: v.replace("/", " of "), [
        ("docs/writeup.md", "the point design classifies {v}"),
        ("docs/site/studies.md", "point design classifies {v}"),
    ]),
    ("ROBUST_robust_correct", lambda v: v.replace("/", " of "), [
        ("docs/writeup.md", "and the ensemble design {v}"),
        ("docs/site/studies.md", "the ensemble design {v}"),
    ]),
    ("E5B_p5_margin_ensemble", lambda v: f"{v:.2f}", [
        ("docs/site/studies.md", "| ensemble-refined | 200/200 (98\u2013100%) | +{v} "),
    ]),
]


@pytest.mark.parametrize("key,fmt,sites", QUOTED)
def test_prose_quotes_current_value(key, fmt, sites):
    """A headline number still appears inside the claim that states it."""
    r = json.loads(RESULTS_JSON.read_text())
    assert key in r, f"{key} missing from results.json"
    wanted = fmt(r[key])
    for rel, template in sites:
        phrase = template.replace("{v}", wanted)
        text = (ROOT / rel).read_text()
        assert phrase in text, (
            f"{rel} no longer contains {phrase!r}. Either the artifact moved "
            f"and the prose was not updated, or the sentence was reworded and "
            f"this template needs updating with it."
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
    assert len(set(salt.tolist())) > 1, (
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
    correct = d["gradients"]["jump time closed over differentiably"]
    assert correct, "no gradients recorded for the documented usage"
    assert all(abs(g - d["expected"]) < 1e-4 for g in correct.values()), (
        "Diffrax no longer returns the exact gradient under its documented "
        "usage; docs/site/related.md says it does and must be revisited"
    )


def test_every_committed_artifact_is_documented():
    """docs/site/artifacts.md indexes every artifact, and indexes nothing else.

    The provenance page is only worth reading if it is complete. Adding an
    artifact without a row leaves a published number with no stated source,
    which is the state this repository exists to make impossible.
    """
    pytest.importorskip("docutils")
    sys.path.insert(0, str(ROOT / "docs" / "site" / "_ext"))
    from artifact_index import ARTIFACTS

    tracked = subprocess.run(
        ["git", "ls-files", "experiments", "scripts"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    on_disk = {p for p in tracked if p.endswith((".npz", ".npy", ".json"))}
    indexed = {a for a, _scripts, _backs in ARTIFACTS}

    assert not (on_disk - indexed), (
        f"artifacts with no row in docs/site/artifacts.md: "
        f"{sorted(on_disk - indexed)}"
    )
    assert not (indexed - on_disk), (
        f"artifacts.md lists files that are not committed: "
        f"{sorted(indexed - on_disk)}"
    )

    for artifact, scripts, _backs in ARTIFACTS:
        for rel in [artifact, *scripts]:
            assert (ROOT / rel).exists(), f"{rel} is indexed but missing"


def test_landing_page_timing_matches_the_measurement():
    """The run time on the landing page is the one time_checks.py measured.

    An earlier version of this compared results.json to timing.json and never
    touched the page, so replacing the substitution in conf.py with a literal
    would have passed. Both of the page's readers are exercised here: the
    prose substitution and the stat card.
    """
    timing = json.loads((ROOT / "experiments" / "timing.json").read_text())
    r = json.loads(RESULTS_JSON.read_text())
    measured = timing["total_median_s"]
    assert r["TIMING_checks_total_s"] == measured, (
        "results.json disagrees with timing.json; run collect_results.py"
    )

    site = ROOT / "docs" / "site"
    conf = {"__file__": str(site / "conf.py")}
    exec(compile((site / "conf.py").read_text(), "conf.py", "exec"), conf)
    substitution = conf["myst_substitutions"]["checks_walltime"]
    assert substitution == f"about {measured:.0f} seconds", (
        f"docs/site/conf.py publishes {substitution!r}, not the measured "
        f"{measured} seconds. A substring check on the digits alone passed "
        f"for 'about 25 minutes' against 24.8 seconds, so the unit is part "
        "of what is compared here."
    )

    pytest.importorskip("docutils")
    sys.path.insert(0, str(site / "_ext"))
    from stat_cards import CARDS

    rendered = None
    for keys, fmt, _caption in CARDS:
        if keys == ("TIMING_checks_total_s",):
            rendered = fmt(*(r[k] for k in keys))
    assert rendered is not None, (
        "the landing page has no card reading TIMING_checks_total_s"
    )
    assert rendered == f"{measured:.0f} s", (
        f"the run-time card renders {rendered!r}, not the measured "
        f"{measured} s; replacing the card formatter with a literal passed "
        "when this only checked that the key name appeared in the table"
    )

    page = (site / "index.md").read_text()
    assert "{{ checks_walltime }}" in page, (
        "docs/site/index.md no longer substitutes the measured run time; a "
        "literal there would leave conf.py correct and the page wrong, which "
        "is what this test is named for"
    )

    assert measured < 120, (
        "the four checks now take over two minutes; the landing page reads "
        "the artifact, but getting-started still promises a warm run in seconds"
    )


def test_landing_page_snippet_output_matches_the_solver():
    """The gradient printed in the landing page snippet is the measured one.

    Seventeen digits are typed into docs/site/index.md as the second snippet's
    output. It is correct, but nothing regenerated or checked it, so it could
    outlive the number it quotes.
    """
    import numpy as np

    rows = np.load(ROOT / "experiments" / "e3_rows.npy")
    at_1e3 = rows[int(np.argmin(np.abs(rows[:, 0] - 1e-3)))]
    printed = repr(float(at_1e3[3]))

    page = (ROOT / "docs" / "site" / "index.md").read_text()
    assert printed in page, (
        f"index.md no longer prints {printed}, the saltation gradient the "
        "solver returns at dt = 1e-3; the snippet output was typed and has "
        "drifted from e3_rows.npy"
    )


# A quantity written in a fixed shape, where EVERY occurrence across the docs
# must be one of the currently valid values. The context-anchored guard above
# only proves the right value appears somewhere; it passed while
# docs/writeup.md still carried a superseded "997 of 1000" three sections
# above the corrected one.
SHAPES = [
    (r"(\d{2,4})\s*(?:of|/)\s*1000\b",
     ("ROBUST_point_correct", "ROBUST_robust_correct"),
     "held-out purity over five ensembles"),
    (r"(\d{1,3})\s*(?:of|/)\s*200\b",
     ("E5B_purity_adam", "E5B_purity_cma_es",
      "E5B_purity_nelder_mead", "E5B_purity_ensemble"),
     "held-out purity on one 200-particle draw"),
]

DOC_FILES = [
    "README.md", "docs/writeup.md", "docs/RESULTS.md",
    "docs/site/studies.md", "docs/site/experiments.md",
    "docs/site/index.md", "docs/site/limitations.md",
]


@pytest.mark.parametrize("shape,keys,label", SHAPES)
def test_no_superseded_value_survives_anywhere(shape, keys, label):
    """No page still states an old value of a quantity in a fixed shape."""
    r = json.loads(RESULTS_JSON.read_text())
    valid = set()
    for k in keys:
        if k not in r:
            continue
        v = r[k]
        if isinstance(v, str):
            valid.add(v.split("/")[0])
        else:
            # a purity fraction; the shape publishes it as a count out of 200
            valid.add(str(round(float(v) * 200)))
    if not valid:
        pytest.skip(f"no committed value for {label}")

    stale = []
    for rel in DOC_FILES:
        path = ROOT / rel
        if not path.exists():
            continue
        for i, line in enumerate(path.read_text().splitlines(), 1):
            for found in re.findall(shape, line):
                if found not in valid:
                    stale.append(f"{rel}:{i} states {found}, current are {sorted(valid)}")
    assert not stale, (
        f"superseded values for {label} still published:\n  " + "\n  ".join(stale)
    )


def test_every_committed_figure_has_a_source():
    """figure_source indexes every committed figure, not just the site's.

    The mapping covered 10 of the 12 PNGs; the two it missed are both
    displayed in docs/writeup.md, so the claim that every figure names its
    artifact and its script was true of the site and false of the write-up.
    """
    pytest.importorskip("docutils")
    sys.path.insert(0, str(ROOT / "docs" / "site" / "_ext"))
    from figure_source import SOURCES

    on_disk = {p.name for p in (ROOT / "docs" / "figures").glob("*.png")}
    missing = sorted(on_disk - set(SOURCES))
    assert not missing, f"figures with no source mapping: {missing}"


def test_figures_assert_the_current_numbers():
    """What the figures draw matches the artifacts the prose quotes.

    v0.1.1 shipped three figures whose panel titles still read a withdrawn
    design's numbers while every sentence beside them read the corrected ones.
    Byte-comparing a regenerated PNG catches that but is not portable: the same
    script renders different bytes on Linux and macOS. So the figure scripts
    record the numbers they draw, and those are compared instead.
    """
    r = json.loads(RESULTS_JSON.read_text())
    figs = ROOT / "docs" / "figures"

    claims = json.loads((figs / "rendered_claims.json").read_text())
    rob = claims["study_robustness.png"]
    assert rob["point_correct"] == r["ROBUST_point_correct"], (
        f"study_robustness.png draws point purity {rob['point_correct']}, "
        f"the artifact says {r['ROBUST_point_correct']}; regenerate it"
    )
    assert rob["robust_correct"] == r["ROBUST_robust_correct"], (
        f"study_robustness.png draws ensemble purity {rob['robust_correct']}, "
        f"the artifact says {r['ROBUST_robust_correct']}; regenerate it"
    )

    dc = json.loads((figs / "design_comparison_claims.json").read_text())
    wanted = round(r["E5B_p5_margin_ensemble"], 2)
    assert dc["p5_ensemble"] == wanted, (
        f"design_comparison.png draws {dc['p5_ensemble']} m for the ensemble "
        f"margin, the artifact says {wanted}; regenerate it"
    )


def test_studies_design_table_matches_its_artifact():
    """The four per-design rows are the generated ones.

    Both interval columns were published with nothing computing them, so a
    resample or a design change could not move the page.
    """
    table = json.loads(
        (ROOT / "experiments" / "design_table.json").read_text()
    )["designs"]
    page = (ROOT / "docs" / "site" / "studies.md").read_text()
    for name, d in table.items():
        lo, hi = d["wilson_pct"]
        blo, bhi = d["p5_ci_m"]
        row = (f"| {name} | {d['correct']} ({lo}\u2013{hi}%) | "
               f"{d['p5_margin_m']:+.2f} ({blo:+.2f} to {bhi:+.2f}) |")
        assert row in page, f"studies.md is missing the generated row:\n  {row}"
