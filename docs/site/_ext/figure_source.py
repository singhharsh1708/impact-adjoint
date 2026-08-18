"""Link each figure to the artifact it was plotted from and the script that made it.

Every number on this site is generated from a committed artifact, but a reader
looking at a figure had no path to that artifact. This turns the claim into a
click.

The mapping is stated once, here, and every path in it is checked against the
working tree at build time: a renamed artifact or script fails the build
rather than shipping a dead link.
"""

from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

logger = logging.getLogger(__name__)

BLOB = "https://github.com/singhharsh1708/impact-adjoint/blob/main/"

# figure -> (script, [artifacts it is plotted from])
SOURCES = {
    "e1_trajectory.png": (
        "experiments/make_figures.py",
        ["experiments/e1_history.npy", "experiments/e1_params_history.npy"],
    ),
    "e3_bias.png": ("experiments/make_figures.py", ["experiments/e3_rows.npy"]),
    "e4_sorter.png": ("experiments/make_figures.py", ["experiments/e4_result.npz"]),
    "e6_generalization.png": (
        "experiments/make_figures.py",
        ["experiments/e6_result.npz"],
    ),
    "e5_separator.png": (
        "experiments/make_e5_figure.py",
        ["experiments/e5_result.npz"],
    ),
    "e5b_purity.png": (
        "experiments/make_e5b_figure.py",
        ["experiments/e5b_result.npz"],
    ),
    "design_comparison.png": (
        "experiments/make_design_comparison_figure.py",
        ["experiments/e5b_result.npz", "experiments/e5_result.npz"],
    ),
    "study_verification.png": (
        "experiments/make_study_figures.py",
        [
            "experiments/convergence_result.npz",
            "experiments/gradient_accuracy_result.npz",
            "experiments/scaling_result.npz",
        ],
    ),
    "study_optimizers.png": (
        "experiments/make_study_figures.py",
        ["experiments/optimizer_benchmark.npz"],
    ),
    "study_robustness.png": (
        "experiments/make_study_figures.py",
        [
            "experiments/robustness_stats.npz",
            "experiments/generalization_stats.npz",
        ],
    ),
}


class FigureSource(Directive):
    """`.. figure-source:: <figure filename>`"""

    required_arguments = 1
    has_content = False

    def run(self):
        name = self.arguments[0]
        entry = SOURCES.get(name)
        if entry is None:
            logger.warning("figure-source: no mapping for %s", name)
            return []
        script, artifacts = entry
        root = Path(self.state.document.settings.env.srcdir).parent.parent
        for rel in [script, *artifacts]:
            if not (root / rel).exists():
                logger.warning("figure-source: %s missing for %s", rel, name)
                return []

        para = nodes.paragraph(classes=["ia-figsource"])
        para += nodes.Text("Plotted from ")
        for i, rel in enumerate(artifacts):
            if i:
                para += nodes.Text(" and " if i == len(artifacts) - 1 else ", ")
            leaf = rel.rsplit("/", 1)[-1]
            ref = nodes.reference("", "", refuri=BLOB + rel)
            ref += nodes.literal(text=leaf)
            para += ref
        para += nodes.Text(" by ")
        ref = nodes.reference("", "", refuri=BLOB + script)
        ref += nodes.literal(text=script.rsplit("/", 1)[-1])
        para += ref
        para += nodes.Text(".")
        return [para]


def setup(app):
    app.add_directive("figure-source", FigureSource)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
