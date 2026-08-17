"""Independent flat-terrain closed-form check for the contact solver.

amp = 0, drag = 0: the full multi-bounce trajectory has exact formulas.
  w0  = sqrt(vy0^2 + 2 g y0)                (first impact speed)
  t_1 = (vy0 + w0) / g,  x_1 = vx0 t_1
  after impact k: vx = (1-mu)^k vx0,  vy+ = e^k w0,  flight time T_k = 2 e^k w0 / g
  x_{k+1} = x_k + (1-mu)^k vx0 T_k
Everything below is derived with sympy from these formulas (rational arithmetic,
40-digit evaluation) -- the solver is only called at the very end for comparison.
Also includes a fully hand-derived d x_k / d e (differentiating the recursion by
hand) as a cross-check on the sympy derivation, and an implicit-function-theorem
formula for d x_1 / d amp_i at amp = 0.
"""

import numpy as np
from pathlib import Path

from juliacall import Main as _jl  # boot Julia before anything else, like validate_contact.py

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

import sympy as sp
from tesseract_core import Tesseract


def record(key, value):
    """Append one measurement to experiments/oracle_results.json.

    The headline agreement figures used to be printed and then lost, so the
    numbers quoted in the README and the writeup traced to nothing. This makes
    them artifacts like every other published number.
    """
    import json
    from pathlib import Path as _P

    path = _P(__file__).parent.parent / "experiments" / "oracle_results.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[key] = float(value)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


API_PATH = Path(__file__).parent.parent / "tesseracts" / "contact_sim" / "tesseract_api.py"

# ---------------- symbolic closed form (no solver involvement) ----------------
G = sp.Rational(981, 100)
T_FINAL = sp.Integer(2)
K = 6

vx0, vy0, y0, e, mu = sp.symbols("vx0 vy0 y0 e mu", positive=True)
SUBS = {vx0: 2, vy0: sp.Rational(1, 2), y0: 1, e: sp.Rational(7, 10), mu: sp.Rational(1, 10)}

w0 = sp.sqrt(vy0**2 + 2 * G * y0)
t1 = (vy0 + w0) / G
xs = [vx0 * t1]   # impact positions x_1 .. x_K
ts = [t1]         # impact times   t_1 .. t_K
for k in range(1, K):
    Tk = 2 * e**k * w0 / G
    ts.append(ts[-1] + Tk)
    xs.append(xs[-1] + (1 - mu) ** k * vx0 * Tk)


def val(expr):
    return float(sp.N(expr.subs(SUBS), 40))


ts_num = [val(t) for t in ts]
m = sum(1 for t in ts_num if t <= 2.0)
print("closed-form impact times:", np.round(ts_num, 6).tolist(), "-> impacts before t_final:", m)

# final state on the arc after impact m
dtq = T_FINAL - ts[m - 1]
vxm = (1 - mu) ** m * vx0
vym = e**m * w0
qf_sym = [xs[m - 1] + vxm * dtq, vym * dtq - G * dtq**2 / 2, vxm, vym - G * dtq]

params = [("v0x", vx0), ("v0y", vy0), ("y0", y0), ("e", e), ("mu", mu)]
closed_imp = np.array([val(x) for x in xs[:m]])
closed_qf = np.array([val(q) for q in qf_sym])
closed_dimp = {pn: np.array([val(sp.diff(x, p)) for x in xs[:m]]) for pn, p in params}
closed_dqf = {pn: np.array([val(sp.diff(q, p)) for q in qf_sym]) for pn, p in params}

# hand derivation of d x_k / d e (differentiating the recursion by hand):
#   x_k = x_1 + (2 vx0 w0 / g) * sum_{j=1}^{k-1} ((1-mu) e)^j
#   d x_k / d e = (2 vx0 w0 / g) * sum_{j=1}^{k-1} j (1-mu)^j e^(j-1)
w0n, gn = val(w0), float(G)
pref = 2 * 2.0 * w0n / gn
hand_de = np.array([pref * sum(j * 0.9**j * 0.7 ** (j - 1) for j in range(1, k)) for k in range(1, m + 1)])
print("hand   d x_k/d e:", hand_de)
print("sympy  d x_k/d e:", closed_dimp["e"])
assert np.allclose(hand_de, closed_dimp["e"], rtol=1e-12, atol=1e-12), "hand vs sympy derivation disagree"

# IFT: guard y(t) - a G_i(x(t)) = 0  =>  d t_1/d a_i = -G_i(x_1)/w0 at a=0
# so d x_1 / d amp_i = -vx0 G_i(x_1) / w0
CTR = np.array([1.0, 2.5, 4.0])
WID = np.array([0.5, 0.4, 0.6])
x1n = closed_imp[0]
Gi = np.exp(-((x1n - CTR) ** 2) / (2 * WID**2))
hand_dimp1_damp = -2.0 * Gi / w0n

# ---------------- run the solver and compare ----------------
inputs = {
    "v0": np.array([2.0, 0.5]), "y0": 1.0, "e": 0.7, "mu": 0.1,
    "amp": np.zeros(3), "ctr": CTR, "wid": WID,
    "drag": 0.0, "t_final": 2.0, "dt": 1e-3, "n_samples": 0,
}
t = Tesseract.from_tesseract_api(API_PATH)
res = t.apply(inputs)
nev = int(res["n_events"])
imp = np.asarray(res["impact_x"])[:nev]
qf = np.asarray(res["qf"])
print("solver n_events:", nev)
print("solver impact_x:", imp)
print("closed impact_x:", closed_imp)
print("solver qf:", qf)
print("closed qf:", closed_qf)
assert nev == m, f"event count mismatch: solver {nev} vs closed-form {m}"

traj_err = max(np.max(np.abs(imp - closed_imp)), np.max(np.abs(qf - closed_qf)))
print(f"[trajectory] max abs err vs closed form: {traj_err:.3e}")

jac = t.jacobian(inputs, jac_inputs={"v0", "y0", "e", "mu", "amp"}, jac_outputs={"qf", "impact_x"})

worst = 0.0
for pn, _ in params:
    if pn in ("v0x", "v0y"):
        col = 0 if pn == "v0x" else 1
        aim = np.asarray(jac["impact_x"]["v0"])[:m, col]
        aqf = np.asarray(jac["qf"]["v0"])[:, col]
    else:
        aim = np.asarray(jac["impact_x"][pn])[:m]
        aqf = np.asarray(jac["qf"][pn])
    se_im = np.max(np.abs(aim - closed_dimp[pn]) / (np.abs(closed_dimp[pn]) + 1.0))
    se_qf = np.max(np.abs(aqf - closed_dqf[pn]) / (np.abs(closed_dqf[pn]) + 1.0))
    worst = max(worst, se_im, se_qf)
    print(f"[{pn}] d impact_x/d{pn} solver={np.round(aim, 8).tolist()}")
    print(f"[{pn}]                closed={np.round(closed_dimp[pn], 8).tolist()}  scaled_err={se_im:.2e}")
    print(f"[{pn}] d qf/d{pn} scaled_err={se_qf:.2e}")

a_amp = np.asarray(jac["impact_x"]["amp"])[0]
se_amp = np.max(np.abs(a_amp - hand_dimp1_damp) / (np.abs(hand_dimp1_damp) + 1.0))
worst = max(worst, se_amp)
print(f"[amp] d impact_x1/d amp solver={a_amp}  IFT={hand_dimp1_damp}  scaled_err={se_amp:.2e}")

print(f"\nWORST scaled Jacobian err vs independent closed form: {worst:.3e}")
record("CLOSED_FORM_jacobian_worst", worst)
assert traj_err < 1e-9, f"trajectory vs closed form: {traj_err:.3e} >= 1e-9"
assert worst < 1e-7, f"Jacobian vs closed form: {worst:.3e} >= 1e-7"
print("CLOSED-FORM ORACLE PASSED (trajectory < 1e-9, Jacobian < 1e-7, solver-independent derivation)")
