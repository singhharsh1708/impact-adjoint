#!/usr/bin/env bash
# Second-client demo: the solver's saltation gradients consumed with NOTHING
# but curl: no Python, no JAX, no Julia on the client side. This is the
# "reusable from any client" claim, demonstrated.
#
# Usage: tesseract serve -p 8123 contact-sim   (in another shell), then:
#        ./scripts/second_client_curl.sh
set -euo pipefail
HOST="${1:-http://127.0.0.1:8123}"

INPUTS='{
  "v0": [2.0, 0.5], "y0": 1.0, "e": 0.7, "mu": 0.1,
  "amp": [0.2, 0.1, 0.15], "ctr": [1.0, 2.5, 4.0], "wid": [0.5, 0.4, 0.6],
  "drag": 0.0, "t_final": 2.0, "dt": 0.001, "n_samples": 0
}'

echo "== apply =="
curl -sf -X POST "$HOST/apply" -H 'Content-Type: application/json' \
  -d "{\"inputs\": $INPUTS}" | python3 -m json.tool | head -20

echo
echo "== vector_jacobian_product: d(final x)/d(e, mu) via cotangent on qf[0] =="
curl -sf -X POST "$HOST/vector_jacobian_product" -H 'Content-Type: application/json' \
  -d "{\"inputs\": $INPUTS,
       \"vjp_inputs\": [\"e\", \"mu\"],
       \"vjp_outputs\": [\"qf\"],
       \"cotangent_vector\": {\"qf\": [1.0, 0.0, 0.0, 0.0]}}" | python3 -m json.tool
