#!/usr/bin/env bash
# Copy the shared C++ core from a spicyR checkout (the single source of truth) and record its provenance.
#   tools/sync_core.sh /path/to/spicyR
# Never edit src/core here: change it in spicyR, then sync. tests/test_core_sync.py checks the checksums.
set -euo pipefail
src="${1:?path to a spicyR checkout}"
rm -rf src/core
mkdir -p src/core
cp -r "$src/src/core/include" "$src/src/core/src" src/core/
find src/core -name '*.o' -delete
commit=$(git -C "$src" rev-parse HEAD 2>/dev/null || echo unknown)
{ echo "# spicyR commit $commit"; (cd src/core && find include src -type f | sort | xargs sha256sum); } > src/core/CHECKSUMS
echo "synced core from $src ($commit)"
