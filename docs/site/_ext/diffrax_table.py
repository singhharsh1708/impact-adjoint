"""Render the Diffrax gradient measurement from its committed artifact.

The comparison page makes one quantitative claim about another project's
library. Every other number here is generated from a committed artifact, and
an unflattering claim about somebody else's code is the last place to drop
that standard, so the row is built from
scripts/diffrax_event_gradient.json rather than typed.

The versions are printed alongside the numbers, because this kind of result is
version-specific and a reader needs to know which one was measured.
"""

import json
from pathlib import Path

import docutils.statemachine
from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

logger = logging.getLogger(__name__)
ARTIFACT = "scripts/diffrax_event_gradient.json"


class DiffraxTable(Directive):
    """`.. diffrax-table::` - measured gradients, versions inline."""

    has_content = False

    def run(self):
        root = Path(self.state.document.settings.env.srcdir).parent.parent
        path = root / ARTIFACT
        if not path.exists():
            logger.warning("diffrax-table: %s missing, run the script", ARTIFACT)
            return []
        d = json.loads(path.read_text())
        v = d["versions"]
        solvers = list(next(iter(d["gradients"].values())).keys())

        lines = [
            "```{list-table}",
            ":header-rows: 1",
            f":widths: 34 {' '.join(['22'] * len(solvers))}",
            "",
            "* - Configuration",
        ]
        lines += [f"  - {s}" for s in solvers]
        for label, vals in d["gradients"].items():
            lines.append(f"* - {label}")
            lines += [f"  - `{vals[s]:+.7f}`" for s in solvers]
        lines.append("```")
        lines += [
            "",
            f"Expected "
            f"`{d['expected']:.1f}` throughout. Measured on "
            f"diffrax {v['diffrax']}, jax {v['jax']}, optimistix "
            f"{v['optimistix']}, Python {v['python']}, by "
            f"`scripts/diffrax_event_gradient.py`, which writes "
            f"`{ARTIFACT.split('/')[-1]}`.",
        ]

        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(
            docutils.statemachine.StringList(lines), self.content_offset, node
        )
        return node.children


def setup(app):
    app.add_directive("diffrax-table", DiffraxTable)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
