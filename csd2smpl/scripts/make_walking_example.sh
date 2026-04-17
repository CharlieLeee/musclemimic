#!/usr/bin/env bash
# Produce a walking-specific example end-to-end on the cluster, stopping
# at the point where a local box with a working OpenGL stack can take over
# and render the mp4.
#
# What it does:
#   1. Picks a walking .pred.npz under $PRED_DIR (default: a clean ACCAD forward
#      walk; override by passing a stem relative to $PRED_DIR as argv[1]).
#   2. Repacks it into AMASS schema at $AMASS_DIR/WalkExample/walk.npz.
#   3. Drives examples/retargeting/retarget_visualize.py --no-render to run
#      the SMPL→MyoFullBody shape-fit + motion retarget and populate the
#      musclemimic cache. --no-render sidesteps the missing libEGL.so.0 on
#      shared Jupyter boxes.
#   4. Tars the resulting cache into $CSD_HOME/walking_example.tgz so you
#      can download one file and unpack it locally at ~/.musclemimic.
#
# Usage:
#   bash csd2smpl/scripts/make_walking_example.sh [PRED_STEM]
#
# Examples:
#   # default walking clip
#   bash csd2smpl/scripts/make_walking_example.sh
#
#   # any other stem under $PRED_DIR (no .pred.npz suffix)
#   bash csd2smpl/scripts/make_walking_example.sh ACCAD/Female1Walking_c3d/B10_-_walk_turn_left_\(90\)_stageii
#
# Env overrides (same defaults as run_pipeline.sh):
#   CSD_HOME, AMASS_DIR, PRED_DIR, SMPL_DIR, MUSCLEMIMIC_HOME.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CSD_HOME="${CSD_HOME:-$HOME/csd2smpl}"
AMASS_DIR="${AMASS_DIR:-$CSD_HOME/amass}"
PRED_DIR="${PRED_DIR:-$CSD_HOME/predictions}"
SMPL_DIR="${SMPL_DIR:-$CSD_HOME/body_models/smpl}"

export MUSCLEMIMIC_HOME="${MUSCLEMIMIC_HOME:-$CSD_HOME/.musclemimic}"
export MUSCLEMIMIC_SMPL_MODEL_PATH="${MUSCLEMIMIC_SMPL_MODEL_PATH:-$SMPL_DIR}"

mkdir -p "$MUSCLEMIMIC_HOME"

if [[ -f "$REPO_ROOT/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$REPO_ROOT/.venv/bin/activate"
fi

# A clean, pure forward walk. If it isn't present, the script prints every
# other walking pred.npz under $PRED_DIR so you can pick one.
DEFAULT_STEM="ACCAD/Female1Walking_c3d/B3_-_walk1_stageii"
STEM="${1:-$DEFAULT_STEM}"
PRED_NPZ="$PRED_DIR/${STEM}.pred.npz"

if [[ ! -f "$PRED_NPZ" ]]; then
    echo "[walk] ERROR: $PRED_NPZ not found" >&2
    echo "[walk]   Candidate walking pred.npz under $PRED_DIR:" >&2
    find "$PRED_DIR" -type f -iname '*walk*.pred.npz' | head -20 >&2 || true
    echo "[walk]   Pass a stem relative to $PRED_DIR as argv[1]." >&2
    exit 1
fi
echo "[walk] pred: $PRED_NPZ"

SUBSET="WalkExample"
MOTION_NAME="walk"
MOTION_REL="$SUBSET/$MOTION_NAME"
AMASS_OUT="$AMASS_DIR/$SUBSET/${MOTION_NAME}_poses.npz"

# pred_to_amass appends _poses.npz internally; we pass the subset/motion only.
python -m csd2smpl.scripts.pred_to_amass \
    --pred_npz   "$PRED_NPZ" \
    --amass_root "$AMASS_DIR" \
    --subset     "$SUBSET" \
    --motion     "$MOTION_NAME" \
    --fps        30

if [[ ! -f "$AMASS_OUT" ]]; then
    echo "[walk] ERROR: expected $AMASS_OUT after pred_to_amass" >&2
    exit 1
fi

# Invalidate any prior cache under this motion name so a change of source clip
# produces a matching retargeted trajectory.
CACHE_DIR="$MUSCLEMIMIC_HOME/caches/AMASS/MyoFullBody"
rm -f "$CACHE_DIR/$SUBSET/${MOTION_NAME}_poses.npz" \
      "$CACHE_DIR/$SUBSET/${MOTION_NAME}_poses_analysis.npz"

# Run the retargeter with --no-render: the env-creation step inside
# retarget_visualize.py triggers load_retargeted_amass_trajectory(), which
# writes the cache before any render would start, so we never touch OpenGL.
AMASS_PATH="$AMASS_DIR" \
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
tar czf "$OUT" -C "$MUSCLEMIMIC_HOME" caches/AMASS/MyoFullBody
ls -lh "$OUT"

cat <<EOF

[walk] Cluster work done. Download $OUT to your local box, then:

  cd <repo>
  tar xzf ~/Downloads/$(basename "$OUT") -C ~/.musclemimic
  rm -rf csd2smpl/examples/_mujoco_raw

  .venv/Scripts/python.exe examples/retargeting/retarget_visualize.py \\
      --motion "${MOTION_REL}_poses" \\
      --record --n-episodes 1 --n-steps 300 \\
      --output-dir csd2smpl/examples/_mujoco_raw \\
      --video-name example_walking

  # then (since the cv2 mp4 writes before ffmpeg fails on PATH):
  FF="\$(.venv/Scripts/python.exe -c 'import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())')"
  SRC=\$(find csd2smpl/examples/_mujoco_raw -name 'example_walking*.mp4' -print -quit)
  "\$FF" -y -v error -i "\$SRC" -c:v libx264 -profile:v baseline -preset fast \\
         -crf 23 -an -r 100 csd2smpl/examples/example_walking.mp4
  rm -rf csd2smpl/examples/_mujoco_raw
EOF
