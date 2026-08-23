"""Record the result of Tesseract's own gradient checker as an artifact.

`tesseract run contact-sim check-gradients` was the one number on the site
with no committed artifact behind it: it lived only in terminal output, so
"0 failures / 1574 checks" could not be verified without Docker and a rebuild.

This runs the checker, parses its summary lines, and writes
experiments/check_gradients.json so collect_results.py can publish it like
every other number. Requires the contact-sim image to be built.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
PAYLOAD = ROOT / "tesseracts" / "contact_sim" / "check_payload.json"
LINE = re.compile(r"Gradient check for (\w+) (passed|failed).*?\((\d+) failures? / (\d+) checks?\)")


# The CLI defaults are rtol=0.1 and a sampling budget that draws with
# replacement, so a "1574 checks" headline is mostly cache hits on a few dozen
# distinct comparisons, compared at ten percent. That measures the sampler, not
# the gradient. These settings compare distinct entries at a tolerance the
# oracles say is meaningful: the solver passes at 1e-4 and the finite
# difference itself stops resolving below that.
RTOL = "1e-4"
EPS = "1e-6"
MAX_EVALS = "30"


def _run(rtol):
    """Invoke the checker at one tolerance and return its combined output."""
    env = {
        "TESSERACT_RUNTIME_CHECK_GRADIENTS_RTOL": rtol,
        "TESSERACT_RUNTIME_CHECK_GRADIENTS_EPS": EPS,
        "TESSERACT_RUNTIME_CHECK_GRADIENTS_MAX_EVALS": MAX_EVALS,
    }
    cmd = ["tesseract", "run"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["contact-sim", "check-gradients", f"@{PAYLOAD}"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.stdout + proc.stderr


def main():
    env = {
        "TESSERACT_RUNTIME_CHECK_GRADIENTS_RTOL": RTOL,
        "TESSERACT_RUNTIME_CHECK_GRADIENTS_EPS": EPS,
        "TESSERACT_RUNTIME_CHECK_GRADIENTS_MAX_EVALS": MAX_EVALS,
    }
    cmd = ["tesseract", "run"]
    for k, v in env.items():
        cmd += ["-e", f"{k}={v}"]
    cmd += ["contact-sim", "check-gradients", f"@{PAYLOAD}"]
    print(" ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout + proc.stderr

    rows = LINE.findall(out)
    if not rows:
        print(out[-2000:], file=sys.stderr)
        raise SystemExit("could not parse any check-gradients summary line")

    endpoints = {}
    for name, verdict, failures, checks in rows:
        endpoints[name] = {"passed": verdict == "passed",
                           "failures": int(failures), "checks": int(checks)}
        print(f"  {name:26s} {failures} failures / {checks} checks")

    counts = {v["checks"] for v in endpoints.values()}
    if len(counts) != 1:
        raise SystemExit(f"endpoints disagree on check count: {counts}")

    # the tolerance sweep was previously written here as three literals, which
    # published numbers nobody had measured in the artifact whose whole point is
    # that they are measured. Run it.
    sweep = {}
    for rt in ("1e-4", "1e-5", "1e-6", "1e-7"):
        rows_rt = LINE.findall(_run(rt))
        if rows_rt:
            # The checker emits one line per endpoint. Summing failures across
            # them while taking the check count from a single line mixes two
            # populations: it published "22 of 50" for a rate that is 22 of
            # 150, and made the 1e-7 row read 81 failures out of 50 checks.
            per_endpoint = int(rows_rt[0][3])
            sweep[rt] = {"failures": sum(int(f) for _, _, f, _ in rows_rt),
                         "checks": per_endpoint * len(rows_rt),
                         "endpoints": len(rows_rt),
                         "checks_per_endpoint": per_endpoint}
            print(f"  rtol {rt}: {sweep[rt]['failures']} failures / "
                  f"{sweep[rt]['checks']} checks "
                  f"({len(rows_rt)} endpoints x {per_endpoint})")
    failing = [k for k, v in sweep.items() if v["failures"] > 0]

    data = {
        "rtol": float(RTOL),
        "tolerance_sweep": sweep,
        "first_failing_rtol": float(failing[0]) if failing else None,
        "eps": float(EPS),
        "max_evals": int(MAX_EVALS),
        "endpoints": len(endpoints),
        "checks": counts.pop(),
        "failures": sum(v["failures"] for v in endpoints.values()),
        "per_endpoint": endpoints,
    }
    path = ROOT / "experiments" / "check_gradients.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True))
    print(f"\nwrote {path.name}: {data['failures']} failures / {data['checks']} checks "
          f"on {data['endpoints']} endpoints")

    assert data["failures"] == 0, f"{data['failures']} gradient checks failed"


if __name__ == "__main__":
    main()
