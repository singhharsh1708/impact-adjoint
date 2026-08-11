"""CMA-ES tuning grid for the E5 head-to-head: 3 seeds x 3 sigma0 at the
identical budget and accounting. Guards the headline against the 'you only
ran one CMA config' objection; results are cited in the writeup."""

from pathlib import Path

import numpy as np

from juliacall import Main as _jl

_jl.seval('import Pkg; haskey(Pkg.project().dependencies, "ForwardDiff") || Pkg.add(Pkg.PackageSpec(name="ForwardDiff", version="1.4.5"))')

import cma
from tesseract_core import Tesseract

import sys

sys.path.insert(0, str(Path(__file__).parent))
from e5_separator import AMP0, AMP_MAX, BUDGET, make_objective

ROOT = Path(__file__).parent.parent


def run_one(sim, score, sigma0, seed):
    loss_np, _, counter = make_objective(sim, score)
    best = np.inf
    es = cma.CMAEvolutionStrategy(AMP0.copy(), sigma0, {"bounds": [0.0, AMP_MAX], "verbose": -9, "seed": seed})
    while counter["evals"] + 2 * es.popsize <= BUDGET:
        xs = es.ask()
        vals = [loss_np(x) for x in xs]
        best = min(best, min(vals))
        es.tell(xs, vals)
    return best


def main():
    sim = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "contact_sim" / "tesseract_api.py")
    score = Tesseract.from_tesseract_api(ROOT / "tesseracts" / "score_target" / "tesseract_api.py")

    grid = {}
    for sigma0 in (0.02, 0.05, 0.1):
        for seed in (3, 4, 5):
            b = run_one(sim, score, sigma0, seed)
            grid[(sigma0, seed)] = b
            print(f"sigma0={sigma0:5.2f} seed={seed}  best={b:.4e}")

    vals = np.array(list(grid.values()))
    print(f"\nmedian {np.median(vals):.3e}   best {vals.min():.3e}   worst {vals.max():.3e}")
    np.savez(ROOT / "experiments" / "e5_cma_grid.npz",
             sigma0=np.array([k[0] for k in grid]), seed=np.array([k[1] for k in grid]),
             best=vals)


if __name__ == "__main__":
    main()
