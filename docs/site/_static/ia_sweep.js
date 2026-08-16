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
  };

  const exp = (x) => {
    const s = x.toExponential(2).split("e");
    return s[0] + "e" + (s[1][0] === "-" ? "-" : "") + Math.abs(+s[1]);
  };
  const fixed = (x) => (Object.is(x, 0) ? "0.0" : x.toFixed(7));

  function show(i) {
    const [dt, grid, interp] = rows[i];
    out.dt.textContent = exp(dt);
    out.grid.textContent = fixed(grid);
    out.interp.textContent = fixed(interp);
    out.salt.textContent = fixed(data.truth);
    range.setAttribute(
      "aria-valuetext",
      "dt " + exp(dt) + ", grid reset " + fixed(grid) +
        ", interpolated event " + fixed(interp) +
        ", saltation " + fixed(data.truth)
    );
  }

  range.addEventListener("input", () => show(+range.value));
  show(+range.value);
  root.classList.add("is-live");
})();
