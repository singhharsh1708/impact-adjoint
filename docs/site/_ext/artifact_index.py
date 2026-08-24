"""Index every committed artifact against the script that writes it.

The site's standing claim is that no published number is typed: each one is
read out of an artifact in the repository. A reader had no way to check that
claim in one place. Figures already carry their provenance, but the artifacts
themselves were only discoverable by reading the scripts.

Every path below is checked against the working tree at build time, so a
renamed artifact or script fails the build instead of shipping a table that
quietly describes files that are no longer there.
"""

from pathlib import Path

import docutils.statemachine
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

logger = logging.getLogger(__name__)

BLOB = "https://github.com/singhharsh1708/impact-adjoint/blob/main/"

# artifact -> (producing scripts, what the artifact backs)
ARTIFACTS = [
    ("experiments/e3_rows.npy", ["experiments/e3_naive_vs_saltation.py"],
     "The grid-reset gradient at every step size, next to the saltation one. "
     "This is the artifact behind the landing page's headline failure."),
    ("experiments/convergence_result.npz", ["experiments/study_convergence.py"],
     "Observed order of accuracy against the analytic free-flight solution."),
    ("experiments/oracle_results.json",
     ["scripts/validate_closed_form.py", "scripts/validate_contact.py",
      "scripts/validate_reference.py"],
     "Two solver-independent oracles, a symbolic multi-bounce closed form and "
     "a scipy reimplementation of the Jacobian, plus a finite-difference and "
     "robustness gate through the solver itself."),
    ("experiments/gradient_accuracy_result.npz",
     ["experiments/study_gradient_accuracy.py"],
     "Agreement between the analytic Jacobian and finite differences, swept "
     "over probe size."),
    ("experiments/check_gradients.json", ["scripts/capture_check_gradients.py"],
     "Tesseract's own gradient checker: failures, checks and endpoints, plus "
     "the tolerance sweep from 1e-4 to 1e-7. Needs the contact-sim image."),
    ("experiments/e1_history.npy", ["experiments/e1_inverse_design.py"],
     "E1 miss distance per Adam step, the 1.12 m to 2.7 cm trace."),
    ("experiments/e1_params_history.npy", ["experiments/e1_inverse_design.py"],
     "The E1 parameter path, including the bounce-count crossings."),
    ("experiments/e2_result.npz", ["experiments/e2_calibration.py"],
     "E2 point calibration of restitution and tangential retention."),
    ("experiments/e2b_posterior.npz", ["experiments/e2b_bayesian.py"],
     "NUTS posterior draws with divergences, split R-hat and effective sample "
     "size. The chains run over HTTP against the container."),
    ("experiments/e4_result.npz", ["experiments/e4_terrain_design.py"],
     "E4 terrain that routes different inlet speeds to different cups."),
    ("experiments/e5_result.npz", ["experiments/e5_separator.py"],
     "The 24-parameter resilience separator and its landing errors."),
    ("experiments/e5_cma_grid.npz", ["experiments/e5_cma_grid.py"],
     "The CMA-ES tuning grid, so the baseline is tuned rather than assumed."),
    ("experiments/e5b_result.npz", ["experiments/e5b_robust_separator.py"],
     "Held-out purity and fifth-percentile margins for all four designs, "
     "including the two gradient-free ones."),
    ("experiments/e6_result.npz", ["experiments/e6_generalization.py"],
     "Zero-shot generalization of the design across restitution."),
    ("experiments/generalization_stats.npz",
     ["experiments/study_generalization_stats.py"],
     "Decisiveness of the design under jitter, per restitution."),
    ("experiments/robustness_stats.npz", ["experiments/study_robustness_stats.py"],
     "Wilson intervals, exact McNemar and the paired bootstrap over particle "
     "indices."),
    ("experiments/optimizer_benchmark.npz", ["experiments/study_optimizers.py"],
     "Adam against CMA-ES and Nelder-Mead over five seeds, under both "
     "evaluation-count and wall-clock accounting."),
    ("experiments/scaling_result.npz", ["experiments/study_scaling.py"],
     "Cost per parameter and the reverse-over-forward ratio, up to 581 "
     "parameters."),
    ("experiments/scaling_repeats.json", ["experiments/study_scaling.py"],
     "The same study measured twice, so the run-to-run spread on the slope is "
     "a recorded number rather than a claim."),
    ("experiments/design_table.json", ["experiments/study_design_table.py"],
     "Per-design purity and fifth-percentile margin with Wilson and bootstrap "
     "intervals, for the four-row table on the studies page."),
    ("experiments/timing.json", ["scripts/time_checks.py"],
     "How long the four documented checks take on a warm depot."),
    ("scripts/diffrax_event_gradient.json", ["scripts/diffrax_event_gradient.py"],
     "The measured Diffrax comparison, including the two caller mistakes that "
     "an earlier version of this site mistook for a Diffrax defect."),
    ("experiments/results.json", ["experiments/collect_results.py"],
     "Every published number, collected from the artifacts above. The drift "
     "test asserts that the prose still quotes what this file says."),
]


class ArtifactIndex(Directive):
    """`.. artifact-index::` - artifacts, producers and what they back."""

    has_content = False

    def run(self):
        root = Path(self.state.document.settings.env.srcdir).parent.parent

        lines = ["```{list-table}", ":header-rows: 1", ":widths: 26 26 48", "",
                 "* - Artifact", "  - Written by", "  - What it backs"]
        for artifact, scripts, backs in ARTIFACTS:
            for rel in [artifact, *scripts]:
                if not (root / rel).exists():
                    logger.warning("artifact-index: %s is missing", rel)
            lines.append(f"* - [`{artifact}`]({BLOB}{artifact})")
            lines.append(
                "  - " + ", ".join(
                    f"[`{Path(s).name}`]({BLOB}{s})" for s in scripts
                )
            )
            lines.append(f"  - {backs}")
        lines.append("```")

        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(
            docutils.statemachine.StringList(lines), self.content_offset, node
        )
        return node.children


def setup(app):
    app.add_directive("artifact-index", ArtifactIndex)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
