// Step-size sweep readout. All values come from a JSON block inlined at build
// time out of experiments/e3_rows.npy; nothing is computed here.
(() => {
  "use strict";
  const root = document.getElementById("ia-sweep");
  if (!root) return;

  const store = document.getElementById("ia-sweep-data");
  const range = document.getElementById("ia-sweep-range");
  if (!store || !range) return;

  let data;
  try {
    data = JSON.parse(store.textContent);
  } catch (e) {
    return; // leave the noscript table as the record
  }
  const rows = data.rows || [];
  if (!rows.length) return;

  const out = {
    dt: root.querySelector("[data-dt]"),
    grid: root.querySelector("[data-grid]"),
    interp: root.querySelector("[data-interp]"),
    salt: root.querySelector("[data-salt]"),
    saltDelta: root.querySelector("[data-saltdelta]"),
  };

  // keep the exponent sign, and match the server-rendered fallback table
  const exp = (x) => x.toExponential(2).replace("e+", "e");
  const fixed = (x) => (Object.is(x, 0) ? "0.0" : x.toFixed(7));

  function show(i) {
    const [dt, grid, interp, salt] = rows[i];
    out.dt.textContent = exp(dt);
    out.grid.textContent = fixed(grid);
    out.interp.textContent = fixed(interp);
    out.salt.textContent = fixed(salt);
    // the readout agrees to seven decimals at every step size, so show the
    // residual against the finest solve: without it a reader cannot tell a
    // per-row measurement from a printed constant
    const ref = rows[rows.length - 1][3];
    const d = salt - ref;
    out.saltDelta.textContent =
      d === 0 ? "= finest dt" : (d > 0 ? "+" : "") + d.toExponential(1) + " vs finest dt";
    range.setAttribute(
      "aria-valuetext",
      "dt " + exp(dt) + ", grid reset " + fixed(grid) +
        ", interpolated event " + fixed(interp) +
        ", saltation " + fixed(salt)
    );
  }

  range.addEventListener("input", () => show(+range.value));
  show(+range.value);
  root.classList.add("is-live");
})();
