#!/usr/bin/env bash
# Clone RPent at f70d720 and apply the paper's patch plus plan C's oracle patch (research/c/README.md, step 2).
#   research/c/setup_rpent.sh <destination dir>
# Then install it into RPENT_ENV:  $RPENT_ENV/bin/pip install -e "<destination dir>[full]"
set -euo pipefail
DEST=${1:?usage: setup_rpent.sh <destination dir>}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
EXPECTED_TREE=10a16d6ff9c04d3cbf14433ee7fe6728b9ec41e6   # f70d720 + rpent_f70d720.patch + rpent_oracle_c.patch
[ -e "$DEST" ] && { echo "$DEST exists; give a new directory"; exit 1; }
git clone https://github.com/RLinf/RPent.git "$DEST"
cd "$DEST"
git checkout -q f70d720
git apply "$REPO/patches/rpent/rpent_f70d720.patch"
git apply "$REPO/patches/rpent/rpent_oracle_c.patch"
git add -A
tree=$(git write-tree)
if [ "$tree" != "$EXPECTED_TREE" ]; then
  echo "unexpected tree $tree (expected $EXPECTED_TREE)"; exit 1
fi
echo "RPent with both patches at $DEST (tree $tree)"
echo "next: \$RPENT_ENV/bin/pip install -e \"$DEST[full]\" && liberopro-download-assets --skip-existing"
