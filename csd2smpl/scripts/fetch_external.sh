#!/usr/bin/env bash
# Fetch external assets used by csd2smpl into csd2smpl/data/external/.
#
# This repo does not redistribute these files (MPG license forbids it);
# we pull them from their authoritative published locations. The user
# must have already accepted the AMASS / SMPL license elsewhere to use
# them in research workflows.
#
# Idempotent — re-running skips files that are already present and non-empty.
#
# Usage:
#   bash csd2smpl/scripts/fetch_external.sh [dest]
#
# Env:
#   FORCE=1   re-download even if the file exists

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${1:-$REPO_ROOT/csd2smpl/data/external}"
FORCE="${FORCE:-0}"

mkdir -p "$DEST"

# name → URL list. Add new assets here.
declare -A ASSETS=(
    ["ssm_all_marker_placements.json"]="https://raw.githubusercontent.com/nghorbani/amass/master/src/amass/data/ssm_all_marker_placements.json"
)

fetch_one() {
    local name="$1"
    local url="$2"
    local out="$DEST/$name"

    if [[ "$FORCE" != "1" && -s "$out" ]]; then
        echo "[fetch] $name already present at $out (skip)"
        return 0
    fi

    echo "[fetch] $name  <-  $url"
    local tmp="${out}.tmp.$$"
    if command -v curl >/dev/null 2>&1; then
        curl -fsSL "$url" -o "$tmp"
    elif command -v wget >/dev/null 2>&1; then
        wget -q "$url" -O "$tmp"
    else
        echo "error: neither curl nor wget found on PATH" >&2
        rm -f "$tmp"
        exit 1
    fi

    # Trivial content check: non-empty + parseable as JSON for .json files.
    if [[ ! -s "$tmp" ]]; then
        echo "error: downloaded $name is empty" >&2
        rm -f "$tmp"
        exit 1
    fi
    if [[ "$name" == *.json ]] && command -v python3 >/dev/null 2>&1; then
        python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$tmp" \
            || { echo "error: $name is not valid JSON" >&2; rm -f "$tmp"; exit 1; }
    fi
    mv "$tmp" "$out"
    local size
    size=$(wc -c < "$out")
    echo "        ${size} bytes"
}

for name in "${!ASSETS[@]}"; do
    fetch_one "$name" "${ASSETS[$name]}"
done

echo ""
echo "External assets under: $DEST"
ls -la "$DEST" | tail -n +2
