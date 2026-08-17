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


def main():
    cmd = ["tesseract", "run", "contact-sim", "check-gradients", f"@{PAYLOAD}"]
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

    data = {
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
