# csd2smpl → musclemimic example

End-to-end walk-through that turns **one** csd2smpl prediction into both a
raw SMPL skeleton video and a musclemimic mujoco video. The two renderers
are pipeline steps (`visualize`, `render_mujoco`) in
`csd2smpl/scripts/run_pipeline.sh`, so regeneration is a single command.

## What's committed here

| File | Source |
| --- | --- |
| `example.pred.npz` | one `*.pred.npz` from a training run (auto-seeded) |
| `example.png` | frame 0 of the SMPL-24 skeleton render |
| `example.mp4` | 10 s SMPL skeleton animation |
| `example_muscle.mp4` | musclemimic mujoco playback (MyoFullBody retarget) |

Videos are capped short (≤10 s @ 30 fps) so they stay under GitHub's 50 MB
limit. If the mujoco mp4 still exceeds ~25 MB, drop `--n-steps` in the
`render_mujoco` step of the pipeline script.

## One-shot regeneration (recommended)

```bash
bash csd2smpl/scripts/run_pipeline.sh --only visualize
bash csd2smpl/scripts/run_pipeline.sh --only render_mujoco
```

Both steps are idempotent — they skip if the output files already exist.
Delete the artefact and rerun to regenerate.

## Running the two renderers manually

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
python examples/retargeting/retarget_visualize.py \
    --motion "CsdPred/example_poses" --record \
    --n-episodes 1 --n-steps 300 \
    --output-dir csd2smpl/examples/_mujoco_raw \
    --video-name example_muscle
cp csd2smpl/examples/_mujoco_raw/*/example_muscle*.mp4 \
   csd2smpl/examples/example_muscle.mp4
rm -rf csd2smpl/examples/_mujoco_raw
```

## Commit + view locally

```bash
ls -lh csd2smpl/examples/
git add csd2smpl/examples/example.pred.npz \
        csd2smpl/examples/example.png \
        csd2smpl/examples/example.mp4 \
        csd2smpl/examples/example_muscle.mp4
git commit -m "example: add one csd2smpl prediction + SMPL and mujoco videos"
git push
```

On your laptop: `git pull`, open the `.mp4`s in any player.
