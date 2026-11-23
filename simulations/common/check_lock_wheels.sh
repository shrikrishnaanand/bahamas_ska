#!/bin/bash
# Check that every package pinned in uv.lock (main dependencies, not the `test` extra) installs from a prebuilt
# wheel, without compiling, on an old-glibc Linux cluster (manylinux2014 = glibc 2.17), for each supported Python.
# Run after every `uv lock`. A failure names the package: cap it in [tool.uv] constraint-dependencies of
# pyproject.toml at its newest release with a manylinux2014 wheel, and relock.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
uv export --frozen --no-emit-project --no-hashes --quiet -o "$tmp/lock.txt"
status=0
for py in 3.10 3.11 3.12 3.13; do
  if uv pip compile "$tmp/lock.txt" --python-version "$py" --python-platform x86_64-manylinux_2_17 \
       --only-binary :all: --no-deps --quiet -o "$tmp/out.txt"; then
    echo "Python $py: OK"
  else
    echo "Python $py: FAILED (see above)"; status=1
  fi
done
exit $status
