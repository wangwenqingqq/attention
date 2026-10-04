#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p upstream
revision=1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb
if [ ! -e upstream/zoology ]; then
  git clone --no-checkout https://github.com/HazyResearch/zoology.git upstream/zoology
  git -C upstream/zoology checkout --detach "$revision"
fi
[ "$(git -C upstream/zoology rev-parse HEAD)" = "$revision" ] || { echo 'Unexpected Zoology revision' >&2; exit 1; }
git -C upstream/zoology diff --quiet
git -C upstream/zoology diff --cached --quiet
for file in moba_naive.py MOBA_LICENSE; do
  if [ ! -e "upstream/$file" ]; then
    path=moba/moba_naive.py
    [ "$file" != MOBA_LICENSE ] || path=LICENSE
    temporary="$(mktemp upstream/.download.XXXXXX)"
    trap 'rm -f "$temporary"' EXIT
    curl --fail --location "https://raw.githubusercontent.com/MoonshotAI/MoBA/b5d58363311d3ca946f1ec444182727c15e338b5/$path" \
      -o "$temporary"
    mv -n "$temporary" "upstream/$file"
    rm -f "$temporary"
    trap - EXIT
  fi
done
"${ATTENTION_PYTHON:-python3}" - <<'PY'
import hashlib
from pathlib import Path
expected = "508a71e961b82e970e9e9039bee8eb37b8f9b455f98b18d852d27ab69ddd88a1"
assert hashlib.sha256(Path("upstream/moba_naive.py").read_bytes()).hexdigest() == expected, "MoBA reference hash mismatch"
PY
