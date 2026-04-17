#!/usr/bin/env bash
# Produce a walking-specific example end-to-end on the cluster, stopping
# at the point where a local box with a working OpenGL stack can take over
# and render the mp4.
#
# What it does:
#   1. Picks a walking .pred.npz under $PRED_DIR. Resolution order:
#        a. argv[1] — an explicit stem (relative to $PRED_DIR, no .pred.npz).
#        b. default: ACCAD/Female1Walking_c3d/B3_-_walk1_stageii.
#        c. any .pred.npz whose path contains "walk" (case-insensitive).
#        d. any .pred.npz at all (loudly noted — no walking clip was found).
#   2. Repacks to AMASS schema at $AMASS_DIR/WalkExample/walk_poses.npz.
#   3. Drives examples/retargeting/retarget_visualize.py --no-render to
#      populate the musclemimic retarget cache without needing OpenGL.
#   4. Tars the cache into $CSD_HOME/walking_example.tgz for local render.
#
# All stdout + stderr is tee'd to $LOG_DIR/<ts>_make_walking_example.log
# (same pattern as run_pipeline.sh). Each sub-step is introduced so you can
# diff the log against a previous successful run.
#
# Usage:
#   bash csd2smpl/scripts/make_walking_example.sh [PRED_STEM]
#
# Env overrides (same defaults as run_pipeline.sh):
#   CSD_HOME, AMASS_DIR, PRED_DIR, MARKERS_DIR, SMPL_DIR, LOG_DIR,
#   MUSCLEMIMIC_HOME.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CSD_HOME="${CSD_HOME:-$HOME/csd2smpl}"
AMASS_DIR="${AMASS_DIR:-$CSD_HOME/amass}"
PRED_DIR="${PRED_DIR:-$CSD_HOME/predictions}"
MARKERS_DIR="${MARKERS_DIR:-$CSD_HOME/markers}"
SMPL_DIR="${SMPL_DIR:-$CSD_HOME/body_models/smpl}"
LOG_DIR="${LOG_DIR:-$CSD_HOME/logs}"

export MUSCLEMIMIC_HOME="${MUSCLEMIMIC_HOME:-$CSD_HOME/.musclemimic}"
export MUSCLEMIMIC_SMPL_MODEL_PATH="${MUSCLEMIMIC_SMPL_MODEL_PATH:-$SMPL_DIR}"
export PYTHONUNBUFFERED=1

mkdir -p "$MUSCLEMIMIC_HOME" "$LOG_DIR"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

# Tee everything from here on to a timestamped log under $LOG_DIR.
TS="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/${TS}_make_walking_example.log"
exec > >(tee -a "$MAIN_LOG") 2>&1

echo "── [make_walking_example] $(date -u +'%Y-%m-%d %H:%M:%S UTC') ───────────"
echo "    repo     : $REPO_ROOT"
echo "    log      : $MAIN_LOG"
echo "    CSD_HOME : $CSD_HOME"
echo "    AMASS    : $AMASS_DIR"
echo "    PRED     : $PRED_DIR"
echo "    MARKERS  : $MARKERS_DIR"
echo "    SMPL     : $SMPL_DIR"
echo "    MM_HOME  : $MUSCLEMIMIC_HOME"

run_step() {
    local step="$1"; shift
    local log="$LOG_DIR/${TS}_walk_${step}.log"
    echo ""
    echo "── [${step}] $(date -u +'%Y-%m-%d %H:%M:%S UTC') ────────────────"
    echo "    cmd: $*"
    echo "    log: $log"
    if "$@" 2>&1 | tee "$log"; then
        return 0
    else
        local rc=$?
        echo "[${step}] FAILED exit=$rc — see $log" >&2
        exit $rc
    fi
}

# ── Diagnostic snapshot of the pipeline outputs (pre-work) ──
echo ""
echo "── snapshot ─────────────────────────────────────────────"
n_markers=$(find "$MARKERS_DIR" -type f -name '*.markers.npz' 2>/dev/null | wc -l)
n_preds=$(find "$PRED_DIR"    -type f -name '*.pred.npz'    2>/dev/null | wc -l)
echo "    .markers.npz under $MARKERS_DIR : $n_markers"
echo "    .pred.npz    under $PRED_DIR    : $n_preds"
if [[ "$n_preds" -eq 0 ]]; then
    echo "[walk] ERROR: no .pred.npz files exist — run_pipeline.sh predict step never completed." >&2
    echo "[walk]   Run: bash csd2smpl/scripts/run_pipeline.sh --only predict" >&2
    exit 1
fi

echo ""
echo "    first 10 .pred.npz (for reference):"
find "$PRED_DIR" -type f -name '*.pred.npz' | sort | head -10 | sed 's/^/      /'

echo ""
echo "    .pred.npz paths containing 'walk' (case-insensitive):"
mapfile -t WALKS < <(find "$PRED_DIR" -type f -name '*.pred.npz' \
                      | awk 'tolower($0) ~ /walk/ {print}' | sort)
echo "    total: ${#WALKS[@]}"
printf '      %s\n' "${WALKS[@]:0:20}"

# ── Pick the source .pred.npz ──
DEFAULT_STEM="ACCAD/Female1Walking_c3d/B3_-_walk1_stageii"
PRED_NPZ=""

if [[ $# -ge 1 && -n "$1" ]]; then
    PRED_NPZ="$PRED_DIR/${1}.pred.npz"
    echo ""
    echo "[walk] explicit stem → $PRED_NPZ"
    if [[ ! -f "$PRED_NPZ" ]]; then
        echo "[walk] ERROR: explicit stem not found." >&2
        exit 1
    fi
elif [[ -f "$PRED_DIR/$DEFAULT_STEM.pred.npz" ]]; then
    PRED_NPZ="$PRED_DIR/$DEFAULT_STEM.pred.npz"
    echo ""
    echo "[walk] default walking clip found → $PRED_NPZ"
elif [[ "${#WALKS[@]}" -gt 0 ]]; then
    PRED_NPZ="${WALKS[0]}"
    echo ""
    echo "[walk] default not present; auto-picked first walking-match → $PRED_NPZ"
else
    PRED_NPZ=$(find "$PRED_DIR" -type f -name '*.pred.npz' | sort | head -1)
    echo ""
    echo "[walk] WARNING: no 'walk' match found; falling back to first .pred.npz" >&2
    echo "[walk] WARNING: the rendered motion will not be a walk" >&2
    echo "[walk]          → $PRED_NPZ"
fi

SUBSET="WalkExample"
MOTION_NAME="walk"
MOTION_REL="$SUBSET/$MOTION_NAME"
AMASS_OUT="$AMASS_DIR/$SUBSET/${MOTION_NAME}_poses.npz"

# pred_to_amass writes <amass_root>/<subset>/<motion>_poses.npz.
run_step pred_to_amass python -m csd2smpl.scripts.pred_to_amass \
    --pred_npz   "$PRED_NPZ" \
    --amass_root "$AMASS_DIR" \
    --subset     "$SUBSET" \
    --motion     "$MOTION_NAME" \
    --fps        30

if [[ ! -f "$AMASS_OUT" ]]; then
    echo "[walk] ERROR: expected $AMASS_OUT after pred_to_amass" >&2
    exit 1
fi
ls -lh "$AMASS_OUT"

# Invalidate any prior cache under this motion name so a change of source
# clip produces a matching retargeted trajectory.
CACHE_DIR="$MUSCLEMIMIC_HOME/caches/AMASS/MyoFullBody"
rm -f "$CACHE_DIR/$SUBSET/${MOTION_NAME}_poses.npz" \
      "$CACHE_DIR/$SUBSET/${MOTION_NAME}_poses_analysis.npz"

# --no-render means retarget_visualize.py stops right after env creation,
# which is enough to drive load_retargeted_amass_trajectory() and write
# the cache. No OpenGL involved.
run_step retarget env AMASS_PATH="$AMASS_DIR" \
    MUSCLEMIMIC_SMPL_MODEL_PATH="$SMPL_DIR" \
    MUSCLEMIMIC_HOME="$MUSCLEMIMIC_HOME" \
    python "$REPO_ROOT/examples/retargeting/retarget_visualize.py" \
        --motion "${MOTION_REL}_poses" \
        --no-render

TRAJ="$CACHE_DIR/$SUBSET/${MOTION_NAME}_poses.npz"
if [[ ! -f "$TRAJ" ]]; then
    echo "[walk] ERROR: retargeted trajectory not produced at $TRAJ" >&2
    exit 1
fi
ls -lh "$TRAJ"

OUT="$CSD_HOME/walking_example.tgz"
run_step tar tar czf "$OUT" -C "$MUSCLEMIMIC_HOME" caches/AMASS/MyoFullBody
ls -lh "$OUT"

cat <<EOF

[walk] Cluster work done. Source pred:
    $PRED_NPZ

Download $OUT to your local box, then from the repo root:

  tar xzf ~/Downloads/$(basename "$OUT") -C ~/.musclemimic
  rm -rf csd2smpl/examples/_mujoco_raw

  .venv/Scripts/python.exe examples/retargeting/retarget_visualize.py \\
      --motion "${MOTION_REL}_poses" \\
      --record --n-episodes 1 --n-steps 300 \\
      --output-dir csd2smpl/examples/_mujoco_raw \\
      --video-name example_walking

  # cv2 writes the mp4 before the ffmpeg-on-PATH compression step fails, so:
  FF="\$(.venv/Scripts/python.exe -c 'import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())')"
  SRC=\$(find csd2smpl/examples/_mujoco_raw -name 'example_walking*.mp4' -print -quit)
  "\$FF" -y -v error -i "\$SRC" -c:v libx264 -profile:v baseline -preset fast \\
         -crf 23 -an -r 100 csd2smpl/examples/example_walking.mp4
  rm -rf csd2smpl/examples/_mujoco_raw

Full cluster log: $MAIN_LOG
EOF
