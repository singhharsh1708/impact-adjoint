"""Render the landing page's headline figures from experiments/results.json.

These five numbers were hand-typed HTML on the most-read page of the site,
and the drift test that guards quoted numbers only covers README.md,
docs/writeup.md and docs/RESULTS.md. So they were the one place a published
figure could go stale silently, which is the exact failure this project is
about. One of them had: the run time read "~5 min" against a measured 24.8 s,
because the five minutes was the first-run Julia bootstrap rather than the
checks.

Each card below names the results.json key it reads, so a card can only show
what the artifacts say.
"""

import json
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import Directive
from sphinx.util import logging

logger = logging.getLogger(__name__)
ARTIFACT = "experiments/results.json"


def _sci(v):
    mantissa, exponent = f"{v:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


CARDS = [
    (("CONV_order",), lambda v: f"{v:.2f}",
     "observed order of accuracy, against an analytic solution"),
    (("CLOSED_FORM_jacobian_worst",), _sci,
     "multi-bounce agreement with a symbolic closed form"),
    (("CHECKGRAD_failures", "CHECKGRAD_checks"), lambda f, c: f"{int(f)} / {int(c)}",
     "failures per endpoint in Tesseract's gradient checker, at "
     "10<sup>-4</sup> relative tolerance"),
    (("REFERENCE_jacobian_worst",), _sci,
     "analytic Jacobian vs an independent scipy reimplementation"),
    (("TIMING_checks_total_s",), lambda v: f"{v:.0f} s",
     "to re-run the four verification checks, warm"),
]


class StatCards(Directive):
    """`.. stat-cards::` - the landing page figures, read from the artifact."""

    has_content = False

    def run(self):
        root = Path(self.state.document.settings.env.srcdir).parent.parent
        path = root / ARTIFACT
        if not path.exists():
            logger.warning("stat-cards: %s missing, run collect_results.py", ARTIFACT)
            return []
        r = json.loads(path.read_text())

        items = []
        for keys, fmt, caption in CARDS:
            if any(k not in r for k in keys):
                logger.warning("stat-cards: %s missing from %s", keys, ARTIFACT)
                continue
            items.append(
                f"<div><dt>{fmt(*(r[k] for k in keys))}</dt><dd>{caption}</dd></div>"
            )

        html = '<dl class="ia-figures">\n' + "\n".join(items) + "\n</dl>"
        return [nodes.raw("", html, format="html")]


def setup(app):
    app.add_directive("stat-cards", StatCards)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
