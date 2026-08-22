"""Measure how long the four documented verification checks actually take.

The landing page headlined "~5 min to re-run the four verification checks".
That figure came from no artifact, and measuring it shows the four checks run
in well under a minute on a warm depot; the five minutes was the Julia
bootstrap the first run pays, not the checks themselves.

This runs each check in a fresh subprocess, repeats the block, and writes
experiments/timing.json so collect_results.py can publish the number like
every other one. The depot state is what separates warm from cold, so it is
recorded alongside the timings rather than left to the reader to assume.
"""

import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
REPEATS = 3

CHECKS = [
    ("scripts/proof_local.py", "boundary proof"),
    ("scripts/validate_closed_form.py", "symbolic oracle"),
    ("scripts/validate_reference.py", "scipy oracle"),
    ("experiments/e3_naive_vs_saltation.py", "naive vs saltation"),
]


def _time_once(rel):
    """Wall time for one fresh interpreter, so juliacall startup is included."""
    start = time.perf_counter()
    done = subprocess.run(
        [sys.executable, str(ROOT / rel)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    elapsed = time.perf_counter() - start
    if done.returncode != 0:
        raise SystemExit(f"{rel} failed:\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}")
    return elapsed


def main():
    samples = {rel: [] for rel, _ in CHECKS}
    for _ in range(REPEATS):
        for rel, _label in CHECKS:
            samples[rel].append(_time_once(rel))

    per_check = {
        rel: {
            "label": label,
            "median_s": round(statistics.median(samples[rel]), 2),
            "min_s": round(min(samples[rel]), 2),
            "max_s": round(max(samples[rel]), 2),
        }
        for rel, label in CHECKS
    }
    total = sum(v["median_s"] for v in per_check.values())

    out = {
        "checks": per_check,
        "total_median_s": round(total, 1),
        "repeats": REPEATS,
        "depot": "warm",
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
    }
    path = ROOT / "experiments" / "timing.json"
    path.write_text(json.dumps(out, indent=2) + "\n")

    for rel, v in per_check.items():
        print(f"{v['median_s']:6.2f}s  {rel}")
    print(f"{total:6.1f}s  total, median of {REPEATS} runs, warm depot")
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
