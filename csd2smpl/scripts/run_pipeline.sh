#!/usr/bin/env bash
# End-to-end csd2smpl pipeline for the V100 3-way shared cluster.
#
# All artefacts live under $HOME (only /home is persisted on this cluster).
# Each step is idempotent: rerunning skips work that's already complete.
#
# Usage:
#   bash csd2smpl/scripts/run_pipeline.sh [--from STEP] [--only STEP]
#
# Steps (in order): inspect | extract | fetch_external | preflight |
#                   synthesize | train | predict | visualize | render_mujoco
#
# Examples:
#   bash csd2smpl/scripts/run_pipeline.sh                  # full pipeline
#   bash csd2smpl/scripts/run_pipeline.sh --from train     # skip data prep
#   bash csd2smpl/scripts/run_pipeline.sh --only predict
#
# Env vars (override paths if needed):
#   CSD_HOME        default: $HOME/csd2smpl
#   RAW_DIR         default: <repo>/csd2smpl/data/raw
#   AMASS_DIR       default: $CSD_HOME/amass
#   SMPL_DIR        default: $CSD_HOME/body_models/smpl
#   MARKERS_DIR     default: $CSD_HOME/markers
#   CKPT_DIR        default: $CSD_HOME/checkpoints
#   PRED_DIR        default: $CSD_HOME/predictions
#   LOG_DIR         default: $CSD_HOME/logs
#   VENV            default: $REPO/.venv
#   CONFIG          default: csd2smpl/configs/v100_3way.yaml

set -euo pipefail

# ── Path setup ────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CSD_HOME="${CSD_HOME:-$HOME/csd2smpl}"
RAW_DIR="${RAW_DIR:-$REPO_ROOT/csd2smpl/data/raw}"
AMASS_DIR="${AMASS_DIR:-$CSD_HOME/amass}"
SMPL_DIR="${SMPL_DIR:-$CSD_HOME/body_models/smpl}"
MARKERS_DIR="${MARKERS_DIR:-$CSD_HOME/markers}"
CKPT_DIR="${CKPT_DIR:-$CSD_HOME/checkpoints}"
PRED_DIR="${PRED_DIR:-$CSD_HOME/predictions}"
LOG_DIR="${LOG_DIR:-$CSD_HOME/logs}"
VENV="${VENV:-$REPO_ROOT/.venv}"
CONFIG="${CONFIG:-csd2smpl/configs/v100_3way.yaml}"
# Gittable example artefacts produced by the visualize + render_mujoco steps.
EXAMPLE_DIR="${EXAMPLE_DIR:-$REPO_ROOT/csd2smpl/examples}"

# Refuse to write outside /home.
case "$CSD_HOME" in
    /home/*) ;;
    *) echo "error: CSD_HOME must be under /home (got $CSD_HOME)" >&2; exit 1 ;;
esac

mkdir -p "$CSD_HOME" "$AMASS_DIR" "$SMPL_DIR" "$MARKERS_DIR" \
         "$CKPT_DIR" "$PRED_DIR" "$LOG_DIR"

# Reduce CUDA fragmentation in shared scenarios.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

# Line-buffer Python stdout/stderr so `tee` sees progress live, not in chunks.
export PYTHONUNBUFFERED=1

# Activate venv if present (setup_cluster.sh creates it).
if [[ -f "$VENV/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
fi

# ── CLI parsing ───────────────────────────────────────────────
ALL_STEPS=(inspect extract fetch_external preflight synthesize train predict visualize render_mujoco)
FROM_STEP=""
ONLY_STEP=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --from)  FROM_STEP="$2"; shift 2 ;;
        --only)  ONLY_STEP="$2"; shift 2 ;;
        --help|-h)
            sed -n '2,30p' "$0"
            exit 0 ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2 ;;
    esac
done

should_run() {
    local step="$1"
    if [[ -n "$ONLY_STEP" ]]; then
        [[ "$step" == "$ONLY_STEP" ]]
        return $?
    fi
    if [[ -z "$FROM_STEP" ]]; then
        return 0
    fi
    local seen=0
    for s in "${ALL_STEPS[@]}"; do
        [[ "$s" == "$FROM_STEP" ]] && seen=1
        if [[ $seen -eq 1 && "$s" == "$step" ]]; then
            return 0
        fi
    done
    return 1
}

run_log() {
    local step="$1"
    shift
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local log="$LOG_DIR/${ts}_${step}.log"
    echo ""
    echo "── [${step}] $(date -u +'%Y-%m-%d %H:%M:%S UTC') ────────────────"
    echo "    cmd:  $*"
    echo "    log:  $log"
    if "$@" 2>&1 | tee "$log"; then
        return 0
    else
        local rc=$?
        echo "[${step}] FAILED with exit $rc — see $log" >&2
        exit $rc
    fi
}

# ── Step: inspect ─────────────────────────────────────────────
if should_run inspect; then
    run_log inspect python -m csd2smpl.scripts.inspect_raw \
        --raw_dir "$RAW_DIR"
fi

# ── Step: extract ─────────────────────────────────────────────
# Extracts AMASS sub-dataset tarballs into $AMASS_DIR; idempotent.
# Body-model archives are extracted into $SMPL_DIR with their own loop.
if should_run extract; then
    if [[ -n "$(find "$AMASS_DIR" -maxdepth 2 -name '*.npz' -print -quit 2>/dev/null)" ]]; then
        echo "[extract] AMASS already extracted under $AMASS_DIR (skip)"
    else
        run_log extract bash csd2smpl/scripts/download_amass.sh "$RAW_DIR" "$AMASS_DIR"
    fi

    # Body model: look for any tarball/zip whose name hints at SMPL.
    if [[ -n "$(find "$SMPL_DIR" -maxdepth 2 -name '*.pkl' -print -quit 2>/dev/null)" ]]; then
        echo "[extract] SMPL .pkl already present under $SMPL_DIR (skip)"
    else
        echo "[extract] Looking for SMPL body-model archive in $RAW_DIR ..."
        shopt -s nullglob nocaseglob
        body_archives=( "$RAW_DIR"/*smpl*.zip "$RAW_DIR"/*smpl*.tar.bz2 \
                        "$RAW_DIR"/*smpl*.tar.gz "$RAW_DIR"/*smpl*.tar.xz \
                        "$RAW_DIR"/*smpl*.tar    "$RAW_DIR"/*smpl*.tbz2 )
        shopt -u nullglob nocaseglob
        if [[ "${#body_archives[@]}" -eq 0 ]]; then
            echo "[extract] WARNING: no SMPL body-model archive in $RAW_DIR" >&2
            echo "          download from https://smpl.is.tue.mpg.de/ and re-run" >&2
        else
            for body in "${body_archives[@]}"; do
                echo "  unpacking $body -> $SMPL_DIR"
                mkdir -p "$SMPL_DIR"
                case "$body" in
                    *.zip)
                        if command -v unzip >/dev/null 2>&1; then
                            unzip -q -o "$body" -d "$SMPL_DIR"
                        else
                            # Fall back to Python's zipfile module when unzip
                            # isn't installed (common on minimal cluster images).
                            echo "  unzip not found; using python -m zipfile"
                            python -m zipfile -e "$body" "$SMPL_DIR"
                        fi
                        ;;
                    *.tar.bz2|*.tbz2) tar -xjf "$body" -C "$SMPL_DIR" ;;
                    *.tar.gz|*.tgz)   tar -xzf "$body" -C "$SMPL_DIR" ;;
                    *.tar.xz)         tar -xJf "$body" -C "$SMPL_DIR" ;;
                    *.tar)            tar -xf  "$body" -C "$SMPL_DIR" ;;
                esac
                break
            done
        fi
        # Post-extract normalisation: flatten + rename to smplx convention.
        # The SMPL v1.1.0 release zip nests as
        #   SMPL_python_v.1.1.0/smpl/models/basicmodel_{f,m,neutral}_lbs_10_207_0_v1.1.0.pkl
        # but smplx expects SMPL_{FEMALE,MALE,NEUTRAL}.pkl at the top level.
        mapfile -t found_pkls < <(find "$SMPL_DIR" -name '*.pkl' -type f)
        for src in "${found_pkls[@]}"; do
            base="$(basename "$src")"
            case "$base" in
                basicmodel_f_lbs*)       target="$SMPL_DIR/SMPL_FEMALE.pkl" ;;
                basicmodel_m_lbs*)       target="$SMPL_DIR/SMPL_MALE.pkl" ;;
                basicmodel_neutral_lbs*) target="$SMPL_DIR/SMPL_NEUTRAL.pkl" ;;
                SMPL_FEMALE.pkl|SMPL_MALE.pkl|SMPL_NEUTRAL.pkl) continue ;;
                *) continue ;;
            esac
            if [[ ! -f "$target" ]]; then
                echo "  rename $base -> $(basename "$target")"
                mv "$src" "$target"
            fi
        done
        # Remove now-empty nested dirs left behind by the zip layout.
        find "$SMPL_DIR" -mindepth 1 -type d -empty -delete 2>/dev/null || true

        if [[ -z "$(find "$SMPL_DIR" -maxdepth 1 -name 'SMPL_*.pkl' -print -quit 2>/dev/null)" ]]; then
            echo "[extract] WARNING: no SMPL_*.pkl at top of $SMPL_DIR after extraction" >&2
        fi
    fi
fi

# ── Step: fetch_external ──────────────────────────────────────
# Pulls the SSM marker→vertex JSON from its authoritative source
# (not redistributed in this repo due to MPG license).
SSM_JSON="$REPO_ROOT/csd2smpl/data/external/ssm_all_marker_placements.json"
if should_run fetch_external; then
    if [[ -s "$SSM_JSON" ]]; then
        echo "[fetch_external] $SSM_JSON already present (skip)"
    else
        run_log fetch_external bash csd2smpl/scripts/fetch_external.sh
    fi
fi

# ── Step: preflight ───────────────────────────────────────────
if should_run preflight; then
    run_log preflight python -m csd2smpl.scripts.preflight \
        --amass_root "$AMASS_DIR" \
        --smpl_dir   "$SMPL_DIR" \
        --out_root   "$MARKERS_DIR" \
        --require_gpu
fi

# ── Step: synthesize ──────────────────────────────────────────
if should_run synthesize; then
    if [[ -n "$(find "$MARKERS_DIR" -name '*.markers.npz' -print -quit 2>/dev/null)" ]]; then
        echo "[synthesize] markers already present under $MARKERS_DIR (skip; rm to regenerate)"
    else
        run_log synthesize python -m csd2smpl.scripts.synthesize_dataset \
            --amass_root "$AMASS_DIR" \
            --out_root   "$MARKERS_DIR" \
            --model_path "$SMPL_DIR" \
            --layout     cmu_41 \
            --placement  auto \
            --ssm_json   "$SSM_JSON" \
            --target_fps 30
    fi
fi

# ── Step: train ───────────────────────────────────────────────
if should_run train; then
    # Override config paths via env so v100_3way.yaml's ${HOME}-templates resolve.
    export HOME
    if [[ -f "$CKPT_DIR/best.pt" ]]; then
        echo "[train] $CKPT_DIR/best.pt exists (skip; rm to retrain)"
    else
        run_log train python -m csd2smpl.train --config "$CONFIG"
    fi
fi

# ── Step: predict ─────────────────────────────────────────────
# Default to `train` because the stock cluster setup only has ACCAD/BML
# extracted, which are both train-split; the test split ends up empty and
# predict silently writes 0 files. Override with PREDICT_SPLIT=test if you've
# extracted Transitions_mocap / SSM_synced.
PREDICT_SPLIT="${PREDICT_SPLIT:-train}"
if should_run predict; then
    if [[ ! -f "$CKPT_DIR/best.pt" ]]; then
        echo "[predict] no checkpoint at $CKPT_DIR/best.pt — train first" >&2
        exit 1
    fi
    if [[ -n "$(find "$PRED_DIR" -name '*.pred.npz' -print -quit 2>/dev/null)" ]]; then
        echo "[predict] predictions already present under $PRED_DIR (skip; rm to regenerate)"
    else
        run_log predict python -m csd2smpl.predict \
            --config "$CONFIG" \
            --ckpt   "$CKPT_DIR/best.pt" \
            --split  "$PREDICT_SPLIT" \
            --out    "$PRED_DIR"
    fi
fi

# ── Step: visualize ───────────────────────────────────────────
# Render a single prediction as an SMPL-24 skeleton PNG + MP4 for committing
# under csd2smpl/examples/. Uses smplx (already required) + matplotlib +
# imageio; both are in requirements.txt.
EXAMPLE_PRED="$EXAMPLE_DIR/example.pred.npz"
if should_run visualize; then
    if [[ -f "$EXAMPLE_DIR/example.png" && -f "$EXAMPLE_DIR/example.mp4" ]]; then
        echo "[visualize] $EXAMPLE_DIR/example.{png,mp4} exist (skip; rm to regenerate)"
    else
        mkdir -p "$EXAMPLE_DIR"
        if [[ ! -f "$EXAMPLE_PRED" ]]; then
            src="$(find "$PRED_DIR" -name '*.pred.npz' -print -quit 2>/dev/null)"
            if [[ -z "$src" ]]; then
                echo "[visualize] no .pred.npz under $PRED_DIR — run predict first" >&2
                exit 1
            fi
            echo "[visualize] seeding $EXAMPLE_PRED from $src"
            cp "$src" "$EXAMPLE_PRED"
        fi
        run_log visualize python -m csd2smpl.scripts.visualize_pred \
            --pred_npz   "$EXAMPLE_PRED" \
            --smpl_dir   "$SMPL_DIR" \
            --out_dir    "$EXAMPLE_DIR" \
            --max_frames 300 --fps 30
        # visualize_pred names outputs from the input stem; normalise to "example.*".
        stem="$(basename "${EXAMPLE_PRED%.pred.npz}")"
        stem="${stem%.pred}"
        for ext in png mp4; do
            if [[ "$stem" != "example" && -f "$EXAMPLE_DIR/${stem}.${ext}" ]]; then
                mv "$EXAMPLE_DIR/${stem}.${ext}" "$EXAMPLE_DIR/example.${ext}"
            fi
        done
    fi
fi

# ── Step: render_mujoco ───────────────────────────────────────
# Repack the example prediction into AMASS schema, then drive the musclemimic
# MyoFullBody retarget viewer to produce a mujoco mp4. First invocation runs
# the SMPL→muscle-body optimisation (slow; cached to ~/.musclemimic/caches).
if should_run render_mujoco; then
    if [[ -f "$EXAMPLE_DIR/example_muscle.mp4" ]]; then
        echo "[render_mujoco] $EXAMPLE_DIR/example_muscle.mp4 exists (skip; rm to regenerate)"
    else
        if [[ ! -f "$EXAMPLE_PRED" ]]; then
            src="$(find "$PRED_DIR" -name '*.pred.npz' -print -quit 2>/dev/null)"
            if [[ -z "$src" ]]; then
                echo "[render_mujoco] no .pred.npz under $PRED_DIR — run predict first" >&2
                exit 1
            fi
            mkdir -p "$EXAMPLE_DIR"
            echo "[render_mujoco] seeding $EXAMPLE_PRED from $src"
            cp "$src" "$EXAMPLE_PRED"
        fi
        run_log render_mujoco_repack python -m csd2smpl.scripts.pred_to_amass \
            --pred_npz   "$EXAMPLE_PRED" \
            --amass_root "$AMASS_DIR" \
            --subset CsdPred --motion example --fps 30

        raw_out="$EXAMPLE_DIR/_mujoco_raw"
        rm -rf "$raw_out"
        AMASS_PATH="$AMASS_DIR" run_log render_mujoco_viewer \
            python "$REPO_ROOT/examples/retargeting/retarget_visualize.py" \
                --motion "CsdPred/example_poses" \
                --record --n-episodes 1 --n-steps 300 \
                --output-dir "$raw_out" \
                --video-name example_muscle

        # retarget_visualize nests the mp4 under <tag>/<name>.mp4; find and hoist it.
        found="$(find "$raw_out" -name 'example_muscle*.mp4' -print -quit 2>/dev/null)"
        if [[ -z "$found" ]]; then
            echo "[render_mujoco] no mp4 produced under $raw_out" >&2
            exit 1
        fi
        cp "$found" "$EXAMPLE_DIR/example_muscle.mp4"
        rm -rf "$raw_out"
        echo "[render_mujoco] wrote $EXAMPLE_DIR/example_muscle.mp4"
    fi
fi

echo ""
echo "── Pipeline complete ────────────────────────────────────"
echo "  amass     : $AMASS_DIR"
echo "  markers   : $MARKERS_DIR"
echo "  ckpt      : $CKPT_DIR/best.pt"
echo "  preds     : $PRED_DIR"
echo "  examples  : $EXAMPLE_DIR"
echo "  logs      : $LOG_DIR"
