#!/usr/bin/env bash
# Extract AMASS sub-dataset tarballs into the canonical AMASS layout.
#
# AMASS downloads CANNOT be fully scripted: each file requires accepting a
# license at https://amass.is.tue.mpg.de/download.php and clicking through a
# session-cookie-gated link. The recommended workflow is:
#
#   1. Log in at https://amass.is.tue.mpg.de/, accept the license.
#   2. Download every "SMPL-H G" tarball (e.g. CMU.tar.bz2, KIT.tar.bz2, ...)
#      into a single directory on the cluster (~50-100 GB total).
#   3. Run this script to extract them into <dest>/<DatasetName>/...
#
# After extraction, point csd2smpl's synthesize step at <dest>:
#
#   python -m csd2smpl.scripts.synthesize_dataset \
#       --amass_root <dest> \
#       --out_root   <markers_out> \
#       --model_path <smpl_pkl_dir> \
#       --layout     cmu_41 \
#       --target_fps 30
#
# Usage:
#   bash csd2smpl/scripts/download_amass.sh <tarball_dir> <dest_dir>
#
# Optional env vars:
#   AMASS_PARALLEL=N   number of concurrent extractions (default: 4)
#   AMASS_KEEP_TARS=1  do not delete tarballs after successful extraction

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <tarball_dir> <dest_dir>" >&2
    echo "" >&2
    echo "Pre-download all AMASS sub-dataset tarballs (.tar.bz2) into" >&2
    echo "<tarball_dir> from https://amass.is.tue.mpg.de/download.php" >&2
    echo "(SMPL-H G recommended). This script extracts them into <dest_dir>." >&2
    exit 2
fi

TARBALL_DIR="$1"
DEST_DIR="$2"
PARALLEL="${AMASS_PARALLEL:-4}"
KEEP_TARS="${AMASS_KEEP_TARS:-0}"

if [[ ! -d "$TARBALL_DIR" ]]; then
    echo "error: tarball dir not found: $TARBALL_DIR" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"

# Canonical AMASS sub-dataset names (Mahmood et al. 2019).
# Filenames sometimes have version suffixes; the case-insensitive match below
# handles common variants. Add to this list if a new sub-dataset is published.
EXPECTED=(
    ACCAD BMLhandball BMLmovi BioMotionLab_NTroje
    CMU DFaust_67 EKUT Eyes_Japan_Dataset
    HumanEva KIT MPI_HDM05 MPI_Limits MPI_mosh
    SFU SSM_synced TCD_handMocap TotalCapture Transitions_mocap
)

extract_one() {
    local tar="$1"
    local dest="$2"
    local name
    name="$(basename "$tar" | sed -E 's/\.tar\.(bz2|gz|xz)$//I; s/\.tgz$//I; s/\.tbz2$//I')"
    local out="$dest/$name"

    if [[ -d "$out" && -n "$(ls -A "$out" 2>/dev/null || true)" ]]; then
        echo "skip $name (already extracted at $out)"
        return 0
    fi

    echo "extract $name -> $out"
    mkdir -p "$out"
    # AMASS tarballs unpack as <name>/<subject>/*.npz (some include an
    # extra top-level dir). Strip a leading component if present.
    if ! tar -xf "$tar" -C "$out" --strip-components=0 2>/dev/null; then
        echo "  retry $name with --strip-components=1"
        rm -rf "$out" && mkdir -p "$out"
        tar -xf "$tar" -C "$out" --strip-components=1
    fi

    # If extraction produced a single top-level dir matching the name,
    # promote its contents.
    if [[ -d "$out/$name" ]]; then
        shopt -s dotglob
        mv "$out/$name"/* "$out"/ && rmdir "$out/$name"
        shopt -u dotglob
    fi

    # Sanity check: every sub-dataset should contain at least one *_poses.npz
    # OR a *.npz (older subsets sometimes drop the suffix).
    local n_poses
    n_poses=$(find "$out" -name "*_poses.npz" -o -name "*.npz" 2>/dev/null | wc -l)
    if [[ "$n_poses" -eq 0 ]]; then
        echo "  WARNING: no *.npz found under $out" >&2
    else
        echo "  $name: $n_poses sequences"
    fi

    if [[ "$KEEP_TARS" != "1" ]]; then
        rm -f "$tar"
    fi
}

export -f extract_one
export DEST_DIR

# Find all tarballs to process.
mapfile -t TARS < <(find "$TARBALL_DIR" -maxdepth 1 -type f \
    \( -name "*.tar.bz2" -o -name "*.tar.gz" -o -name "*.tar.xz" \
       -o -name "*.tgz" -o -name "*.tbz2" \) | sort)

if [[ "${#TARS[@]}" -eq 0 ]]; then
    echo "error: no .tar.{bz2,gz,xz} files in $TARBALL_DIR" >&2
    exit 1
fi

echo "Found ${#TARS[@]} tarballs in $TARBALL_DIR"
echo "Extracting to $DEST_DIR (parallel=$PARALLEL, keep_tars=$KEEP_TARS)"
echo ""

if command -v xargs >/dev/null && [[ "$PARALLEL" -gt 1 ]]; then
    printf '%s\n' "${TARS[@]}" \
        | xargs -P "$PARALLEL" -I{} bash -c 'extract_one "$@"' _ {} "$DEST_DIR"
else
    for tar in "${TARS[@]}"; do
        extract_one "$tar" "$DEST_DIR"
    done
fi

echo ""
echo "── AMASS extraction summary ─────────────────────────"
total_npz=0
for name in "${EXPECTED[@]}"; do
    if [[ -d "$DEST_DIR/$name" ]]; then
        n=$(find "$DEST_DIR/$name" -name "*.npz" 2>/dev/null | wc -l)
        total_npz=$((total_npz + n))
        printf "  %-25s %5d sequences\n" "$name" "$n"
    fi
done
printf "  %-25s %5d sequences\n" "TOTAL" "$total_npz"
echo "─────────────────────────────────────────────────────"
echo ""
echo "Next: register the path with musclemimic and run synthesis."
echo "  musclemimic-set-amass-path --path $DEST_DIR"
echo "  python -m csd2smpl.scripts.synthesize_dataset \\"
echo "      --amass_root $DEST_DIR \\"
echo "      --out_root   <markers_out_dir> \\"
echo "      --model_path <smpl_pkl_dir> \\"
echo "      --layout     cmu_41 \\"
echo "      --target_fps 30"
