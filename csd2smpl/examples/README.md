# csd2smpl → musclemimic examples

End-to-end walk-throughs that turn csd2smpl predictions into rendered
videos. Two entry points:

| Helper | Where it runs | Produces |
| --- | --- | --- |
| `csd2smpl/scripts/run_pipeline.sh --only visualize` + `... --only render_mujoco` | cluster with OpenGL | `example.png`, `example_muscle.mp4` |
| `csd2smpl/scripts/make_walking_example.sh` (cluster) + local render | cluster + local | `example_walking.mp4` |

The second flow exists because shared Jupyter clusters often lack
`libEGL.so.0` / `libOpenGL.so.0`. `make_walking_example.sh` does
everything up to but not including the MuJoCo render, tars the cache,
and hands off to a local box that already has a working OpenGL stack.

## What's committed here

| File | Source | Clip |
| --- | --- | --- |
| `example.png` | csd2smpl prediction → SMPL-24 skeleton, frame 0 | first training clip alphabetically |
| `example_muscle.mp4` | same prediction → MuscleMimic `MyoFullBody` retarget | 3.0 s, 100 fps |
| `example_walking.mp4` | ACCAD walking pred → `MyoFullBody` retarget | 1.64 s, 100 fps |

Videos are short (≤3 s, H.264 baseline) so they stay well under
GitHub's 50 MB limit. If regenerating and the mp4 exceeds ~25 MB,
drop `--n-steps` in the render command.

## Flow A — `run_pipeline.sh` (full pipeline on cluster)

```bash
bash csd2smpl/scripts/run_pipeline.sh --only visualize
bash csd2smpl/scripts/run_pipeline.sh --only render_mujoco
```

Both steps are idempotent — they skip if the output files already exist.
Delete the artefact and rerun to regenerate. `render_mujoco` needs a
working OpenGL stack on the box (`libEGL.so.0` etc.); on a Jupyter
cluster that lacks it, use Flow B instead.

## Flow B — walking example, cluster + local split

### On the cluster

```bash
bash csd2smpl/scripts/make_walking_example.sh
#   or: bash csd2smpl/scripts/make_walking_example.sh <stem-under-$PRED_DIR>
```

What it does:
1. Picks a walking `.pred.npz` under `$PRED_DIR`. Resolution order is
   explicit argv[1] → default `ACCAD/Female1Walking_c3d/B3_-_walk1_stageii`
   → first path containing `walk` → first `.pred.npz` at all (loudly
   warned — the rendered motion won't be a walk).
2. Repacks to AMASS schema at `$AMASS_DIR/WalkExample/walk_poses.npz`.
3. Runs `examples/retargeting/retarget_visualize.py --no-render`, which
   triggers `load_retargeted_amass_trajectory()` (shape-fit + retarget
   + cache-write) without touching OpenGL.
4. Tars `~/.musclemimic/caches/AMASS/MyoFullBody/` into
   `$CSD_HOME/walking_example.tgz`.

All stdout + stderr is tee'd to `$LOG_DIR/<ts>_make_walking_example.log`,
and each sub-step (`pred_to_amass`, `retarget`, `tar`) also writes its
own per-step log.

### Locally

Download `walking_example.tgz`, then from the repo root:

```bash
tar xzf ~/Downloads/walking_example.tgz -C ~/.musclemimic
rm -rf csd2smpl/examples/_mujoco_raw

.venv/Scripts/python.exe examples/retargeting/retarget_visualize.py \
    --motion "WalkExample/walk_poses" \
    --record --n-episodes 1 --n-steps 300 \
    --output-dir csd2smpl/examples/_mujoco_raw \
    --video-name example_walking

# cv2 writes the mp4 before the ffmpeg-on-PATH compression step fails, so:
FF="$(.venv/Scripts/python.exe -c 'import imageio_ffmpeg as i; print(i.get_ffmpeg_exe())')"
SRC=$(find csd2smpl/examples/_mujoco_raw -name 'example_walking*.mp4' -print -quit)
"$FF" -y -v error -i "$SRC" -c:v libx264 -profile:v baseline -preset fast \
      -crf 23 -an -r 100 csd2smpl/examples/example_walking.mp4
rm -rf csd2smpl/examples/_mujoco_raw
```

On Linux/macOS the venv python is at `.venv/bin/python`.

## Manual renderers (without the helpers)

```bash
# 1) SMPL skeleton (smplx + matplotlib + imageio — no mujoco)
python -m csd2smpl.scripts.visualize_pred \
    --pred_npz csd2smpl/examples/example.pred.npz \
    --smpl_dir $HOME/csd2smpl/body_models/smpl \
    --out_dir  csd2smpl/examples \
    --max_frames 300 --fps 30

# 2) musclemimic mujoco render (slow first run — retarget + cache)
python -m csd2smpl.scripts.pred_to_amass \
    --pred_npz   csd2smpl/examples/example.pred.npz \
    --amass_root $HOME/csd2smpl/amass \
    --subset CsdPred --motion example --fps 30

AMASS_PATH=$HOME/csd2smpl/amass \
MUSCLEMIMIC_SMPL_MODEL_PATH=$HOME/csd2smpl/body_models/smpl \
python examples/retargeting/retarget_visualize.py \
    --motion "CsdPred/example_poses" --record \
    --n-episodes 1 --n-steps 300 \
    --output-dir csd2smpl/examples/_mujoco_raw \
    --video-name example_muscle
cp csd2smpl/examples/_mujoco_raw/*/example_muscle*.mp4 \
   csd2smpl/examples/example_muscle.mp4
rm -rf csd2smpl/examples/_mujoco_raw
```

## Known caveat — `best.pt` on the cluster is an epoch-1 snapshot

`csd2smpl/data/c3d_dataset.py:38-53` uses the paper-canonical AMASS
split: val = `{HumanEva, MPI_HDM05, SFU, MPI_mosh}`. None of those are
extracted on the shared V100 cluster — only `ACCAD` and `BMLhandball`
are. Consequence:

- `train.py:163` divides val loss by `max(len(val_loader), 1)`; with
  an empty val set the value stays at `0.0` every epoch.
- `train.py:169` saves `best.pt` on `val_loss < best_val`. Epoch 1
  wins (`0.0 < inf`); no later epoch beats `0.0 < 0.0`. So the
  checkpoint `predict` loads is essentially an untrained snapshot.

Visually this shows up as mushy, drifting motion in the renders. To
fix, either extract one of the paper's val subsets onto the cluster
or carve a held-out slice from ACCAD by-file inside `c3d_dataset.py`,
then delete `$CKPT_DIR/best.pt` and rerun
`run_pipeline.sh --only train`.
