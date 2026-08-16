#!/usr/bin/env bash
# Build the site from the current commit, then deploy it.
#
# The footer names the commit the site was built from, and that SHA is baked in
# at build time. Building before committing and deploying afterwards therefore
# ships a page that names the wrong commit, which has happened twice. This
# refuses to deploy a dirty tree and always rebuilds first.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "$(git status --porcelain)" ]; then
  echo "working tree is dirty; commit first so the provenance SHA is real" >&2
  git status --short >&2
  exit 1
fi

HEAD_SHA=$(git rev-parse HEAD)
rm -rf docs/site/build
.venv/bin/python -m sphinx -b html docs/site docs/site/build -W -n

BUILT=$(grep -o 'commit/[0-9a-f]\{40\}' docs/site/build/index.html | head -1 | cut -d/ -f2)
if [ "$BUILT" != "$HEAD_SHA" ]; then
  echo "built provenance $BUILT != HEAD $HEAD_SHA" >&2
  exit 1
fi

cd docs/site/build
vercel link --yes --project impact-adjoint >/dev/null
vercel deploy --prod --yes
echo "deployed ${HEAD_SHA:0:7}"
