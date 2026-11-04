#!/usr/bin/env bash
# post_n32.sh : post-completion step for a finished codec bundle (no GPU, no network).
#   bash h100/post_n32.sh BUNDLE MARKER
# 1. h100/codec_verify.py BUNDLE (independent, read-only) -> BUNDLE/codec_verify.json; any non-zero exit stops here and writes no marker
# 2. MARKER written atomically (tmp + mv -f) with the bundle path, only on exit 0
# 3. lists the queues/entries/*.json whose requires_paths contain MARKER: the runner starts them once the marker exists
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
case "${1:-}" in -h|--help|"") sed -n 2,6p "$0"; exit 0;; esac
B=$(realpath "$1"); M=$2
[ -d "$B" ] || { echo "post_n32: no bundle dir $B"; exit 2; }
python3 "$ROOT/h100/codec_verify.py" "$B" > "$B/codec_verify.json.tmp" || { mv -f "$B/codec_verify.json.tmp" "$B/codec_verify.json"; echo "post_n32: codec_verify FAILED, no marker (see $B/codec_verify.json)"; exit 1; }
mv -f "$B/codec_verify.json.tmp" "$B/codec_verify.json"
mkdir -p "$(dirname "$M")"; printf '%s\n' "$B" > "$M.tmp.$$" && mv -f "$M.tmp.$$" "$M"
echo "post_n32: PASS, wrote $M"
python3 - "$ROOT/queues/entries" "$M" <<'PY'
import json, sys
from pathlib import Path
for p in sorted(Path(sys.argv[1]).glob("*.json")):
    e = json.loads(p.read_text())
    if sys.argv[2] in e.get("requires_paths", []):
        print(f"released: {e['id']} ({p.name}) repo_sha {e.get('repo_sha')}")
PY
