"""Emit the step-size sweep widget from the committed E3 artifact.

The central claim on the index page is that grid-reset autodiff returns
exactly 0.0 at every step size. That is a sweep over one scalar, and the sweep
is already committed as experiments/e3_rows.npy, so the page can let a reader
check it themselves instead of asserting it.

Nothing is recomputed here and nothing is rounded into existence: the dt
column and both gradient columns are read straight out of the .npy, and the
saltation value is E3_truth from experiments/results.json, which is the
solver's analytic Jacobian. If a number is not in an artifact it does not
reach the page.

The data is inlined as application/json rather than fetched, so the widget
works from a file:// copy and costs no extra request. A full table of the
same rows is emitted inside <noscript>, so the claim is still checkable with
scripting turned off.
"""

import json
from pathlib import Path

from sphinx.util import logging

logger = logging.getLogger(__name__)

PLACEHOLDER = '<div id="ia-sweep"></div>'


def _load(app):
    root = Path(app.srcdir).parent.parent
    rows = root / "experiments" / "e3_rows.npy"
    results = root / "experiments" / "results.json"
    if not rows.exists() or not results.exists():
        return None
    import numpy as np

    a = np.load(rows)
    truth = json.loads(results.read_text())["E3_truth"]
    if a.shape[1] < 4:
        return None  # pre-saltation artifact: rerun experiments/e3_naive_vs_saltation.py
    salt = a[:, 3]
    return {
        "truth": truth,
        "spread": float(salt.max() - salt.min()),
        # (dt, grid-reset, interpolated-event, saltation) all measured at that dt
        "rows": [[float(r[0]), float(r[1]), float(r[2]), float(r[3])] for r in a],
    }


def _exp(x):
    """Match the JS formatting exactly, so the two agree."""
    s = f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    return s


def _fallback_table(data):
    head = (
        "<table class='docutils align-default'><thead><tr>"
        "<th>step size dt</th><th>grid reset</th>"
        "<th>interpolated event</th><th>saltation</th></tr></thead><tbody>"
    )
    body = "".join(
        f"<tr><td>{_exp(dt)}</td><td>{grid:.7f}</td>"
        f"<td>{interp:.7f}</td><td>{salt:.7f}</td></tr>"
        for dt, grid, interp, salt in data["rows"]
    )
    return head + body + "</tbody></table>"


def _markup(data):
    n = len(data["rows"]) - 1
    return f"""<div id="ia-sweep" class="ia-sweep">
  <script type="application/json" id="ia-sweep-data">{json.dumps(data)}</script>
  <label class="ia-sweep-label" for="ia-sweep-range">
    Step size <code>dt</code>, over the range E3 sweeps
  </label>
  <input id="ia-sweep-range" class="ia-sweep-range" type="range"
         min="0" max="{n}" step="1" value="0"
         aria-describedby="ia-sweep-note" />
  <output class="ia-sweep-out" for="ia-sweep-range" aria-live="polite">
    <span class="ia-sweep-dt">dt = <b data-dt>-</b></span>
    <span class="ia-sweep-cell" data-kind="wrong">
      <span class="ia-sweep-name">grid reset</span>
      <b data-grid>-</b>
    </span>
    <span class="ia-sweep-cell" data-kind="approx">
      <span class="ia-sweep-name">interpolated event</span>
      <b data-interp>-</b>
    </span>
    <span class="ia-sweep-cell" data-kind="exact">
      <span class="ia-sweep-name">saltation</span>
      <b data-salt>-</b>
      <span class="ia-sweep-delta" data-saltdelta>-</span>
    </span>
  </output>
  <p id="ia-sweep-note" class="ia-sweep-note">
    Left is coarse, right is fine: the slider runs from
    <code>{_exp(data['rows'][0][0])}</code> down to
    <code>{_exp(data['rows'][-1][0])}</code>. All three columns are measured at
    the step size shown, including saltation, which is solved afresh at each
    one rather than carried over. Grid reset never leaves zero; the
    interpolated repair wanders; saltation moves by
    <code>{data['spread']:.2e}</code> across the whole sweep, against a
    reference value of <code>{data['truth']}</code>. Read from
    <code>experiments/e3_rows.npy</code>. The sweep is flat-terrain ballistic
    flight, where RK4 is exact by construction; that is what isolates the
    event term from integrator error.
  </p>
  <noscript>{_fallback_table(data)}</noscript>
</div>"""


def on_build_finished(app, exception):
    if exception is not None or app.builder.name != "html":
        return
    data = _load(app)
    if data is None:
        logger.warning("sweep_widget: E3 artifacts missing, widget not emitted")
        return
    page = Path(app.outdir) / "index.html"
    if not page.exists():
        return
    html = page.read_text(encoding="utf-8")
    if 'id="ia-sweep-data"' in html:
        return  # incremental build: this page was not rewritten
    if PLACEHOLDER not in html:
        logger.warning("sweep_widget: placeholder not found on index")
        return
    page.write_text(html.replace(PLACEHOLDER, _markup(data), 1), encoding="utf-8")
    logger.info("sweep_widget: inlined %d sweep rows", len(data["rows"]))


def setup(app):
    app.connect("build-finished", on_build_finished)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
