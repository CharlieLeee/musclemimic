# csd2smpl → musclemimic example

End-to-end walk-through that turns **one** csd2smpl prediction into both a
raw SMPL skeleton video and a musclemimic mujoco video.

## What's committed here

| File | Source |
| --- | --- |
| `example.pred.npz` | one copy of a `*.pred.npz` from a training run |
| `example.png` | frame 0 of the SMPL skeleton render |
| `example.mp4` | full SMPL skeleton animation |
| `example_muscle.mp4` | musclemimic mujoco playback |

Full-resolution mp4s are kept short (≤10 s) so they stay under GitHub's
50 MB hard limit. Regenerate with the commands below if you want the full
sequence at higher dpi.

## 1. Pick a prediction and copy it into the repo

Ran on the cluster (`cd ~/musclemimic`):

```bash
# pick the first prediction file from the train split run
SRC=$(find $HOME/csd2smpl/predictions -name '*.pred.npz' | head -1)
echo "using: $SRC"
cp "$SRC" csd2smpl/examples/example.pred.npz
```

## 2. Render the raw SMPL skeleton (Part 1)

No mujoco, no retargeting — just `smplx` + matplotlib.

```bash
pip install matplotlib "imageio[ffmpeg]>=2.34"   # one-time, if missing

python -m csd2smpl.scripts.visualize_pred \
    --pred_npz csd2smpl/examples/example.pred.npz \
    --smpl_dir $HOME/csd2smpl/body_models/smpl \
    --out_dir  csd2smpl/examples \
    --max_frames 300 --fps 30
```

Outputs `csd2smpl/examples/example.png` + `example.mp4`.

## 3. Render the musclemimic mujoco video (Part 2)

Two steps: repack the prediction into AMASS schema, then call the existing
`retarget_visualize.py` viewer against it.

```bash
# a) write an AMASS-compatible NPZ under the AMASS root
python -m csd2smpl.scripts.pred_to_amass \
    --pred_npz   csd2smpl/examples/example.pred.npz \
    --amass_root $HOME/csd2smpl/amass \
    --subset     CsdPred \
    --motion     example \
    --fps        30

# b) point musclemimic at that AMASS root and render
AMASS_PATH=$HOME/csd2smpl/amass \
python examples/retargeting/retarget_visualize.py \
    --motion "CsdPred/example_poses" \
    --record \
    --n-episodes 1 --n-steps 300 \
    --output-dir csd2smpl/examples/_mujoco_raw \
    --video-name example_muscle

# c) copy the mp4 up one level so git tracks it in a stable path
cp csd2smpl/examples/_mujoco_raw/myofullbody_retargeted/example_muscle.mp4 \
   csd2smpl/examples/example_muscle.mp4
rm -rf csd2smpl/examples/_mujoco_raw
```

First invocation triggers the SMPL→muscle-body retargeting pass (slow — minutes).
Subsequent runs reuse the cache under `$HOME/.musclemimic/caches/`.

## 4. Commit the artifacts

```bash
git add csd2smpl/examples/example.pred.npz \
        csd2smpl/examples/example.png \
        csd2smpl/examples/example.mp4 \
        csd2smpl/examples/example_muscle.mp4
git commit -m "example: add one csd2smpl prediction + SMPL and mujoco videos"
git push
```

## 5. Pull + view locally

```bash
git pull
# open the mp4s in any player; example.png renders in any image viewer
```
