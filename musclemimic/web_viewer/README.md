# Web Viewer

Browser-based visualization tools for motion capture data, SMPL body models, and musculoskeletal simulations.

## Quick Start

```bash
# 1. Get sample C3D data (sparse checkout, not the full 2.2 GB repo)
git clone --filter=blob:none --sparse https://github.com/drivelineresearch/openbiomechanics.git
cd openbiomechanics
git sparse-checkout set baseball_pitching/data/c3d/000002
git sparse-checkout add baseball_hitting/data/c3d/000004
cd ..

# 2. View raw C3D markers
uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer \
  openbiomechanics/baseball_pitching/data/c3d/000002/000002_003034_73_207_002_FF_809.c3d --trail 10

# 3. View on musculoskeletal model (requires SMPL models)
uv run --extra c3d --extra smpl examples/retargeting/retarget_visualize.py \
  --c3d-file openbiomechanics/baseball_pitching/data/c3d/000002/000002_003034_73_207_002_FF_809.c3d
```

Open http://localhost:8080 in your browser after launching any viewer.

---

## 1. C3D Marker Viewer

Interactive 3D viewer for raw C3D motion capture files. No SMPL models required.

### Install

```bash
uv sync --extra c3d
```

### Usage

```bash
# Basic
uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer path/to/file.c3d

# With marker trails and custom port
uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer path/to/file.c3d --trail 10 --port 9090
```

### Examples

```bash
# Pitching (80.9 mph fastball)
uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer \
  openbiomechanics/baseball_pitching/data/c3d/000002/000002_003034_73_207_002_FF_809.c3d

# Hitting with bat tracking (97.2 mph exit velo)
uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer \
  openbiomechanics/baseball_hitting/data/c3d/000004/000004_000103_75_236_R_003_972.c3d --trail 15
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8080 | Server port |
| `--trail` | 0 | Marker trail length in frames (0 = off) |
| `--marker-size` | 0.012 | Marker radius in meters |

### Viewer controls

- **Play/Pause** and **frame slider** for playback
- **Speed**: 0.25x / 0.5x / 1x / 2x
- **Loop** toggle
- **Marker size** and **trail length** sliders
- **Show skeleton / Show markers** toggles

### Color convention

| Color | Meaning |
|-------|---------|
| Blue | Left-side markers/bones |
| Red | Right-side markers/bones |
| Gray | Center markers (head, torso, pelvis) |
| Gold | Bat markers (hitting files, `Marker1`-`Marker10`) |

---

## 2. C3D → SMPL Fitting

Converts C3D marker data to SMPL body model parameters using a MoSh++-inspired surface marker pipeline.

### Install

```bash
uv sync --extra c3d --extra smpl
musclemimic-set-smpl-model-path /path/to/smpl/models   # directory containing SMPLH_NEUTRAL.pkl
```

### How it works

The fitting pipeline (`c3d_to_smpl.py`) implements key ideas from MoSh++ (Mahmood et al., ICCV 2019):

**Surface marker model** -- markers are represented as local-frame coefficients on the SMPL mesh surface, not as joint centers. Each marker is anchored to a specific SMPL vertex (using the exact vertex IDs from the MoSh++ codebase) with a marker-type-dependent skin offset (9.5mm for body markers, 39mm for wrist markers on sticks).

**Two-stage optimization:**

| Stage | What is optimized | Iterations | Purpose |
|-------|-------------------|------------|---------|
| **I** | Body shape (betas), marker surface coefficients, per-frame rigid pose | 320 (4 annealing phases) | Find body proportions and marker positions across ~12 reference frames |
| **II** | Per-frame body pose + translation | 80/frame | Track the motion with velocity-based temporal prior |

**Key features from MoSh++:**
- Marker label canonicalization (Plug-in Gait → MoSh++ canonical names)
- Weight annealing schedule (1.0 → 0.5 → 0.25 → 0.125)
- Missing marker handling (NaN/zero detection, per-frame weight adjustment)
- Procrustes initialization (SVD rigid alignment)
- Surface distance constraints (markers stay at prescribed offset from mesh)
- Velocity prior for temporal smoothness (`pose - 2*prev + prev_prev`)
- Multi-pass first frame with decreasing pose regularization (10x → 5x → 1x)

**Downsampling:** High-frequency captures (e.g. 360 Hz) are automatically downsampled to ~30 Hz for fitting, then the output FPS is reported accordingly.

**Caching:** Results are cached as `.npz` files in `.smpl_cache/` alongside the C3D file. Subsequent runs load from cache.

### Standalone test

```bash
uv run --extra c3d --extra smpl python -c "
import logging; logging.basicConfig(level=logging.INFO, format='%(message)s')
from musclemimic.web_viewer.c3d_to_smpl import fit_smpl_to_c3d_cached
from loco_mujoco.smpl.retargeting import get_smpl_model_path

result = fit_smpl_to_c3d_cached('path/to/file.c3d', get_smpl_model_path())
print(f'pose_aa: {result[\"pose_aa\"].shape}, fps: {result[\"fps\"]}')
"
```

---

## 3. C3D → Musculoskeletal Visualization

End-to-end pipeline: C3D markers → SMPL fitting → retargeting → MyoFullBody/MyoBimanualArm.

### Via retarget_visualize.py (with MuJoCo rendering)

```bash
# Pitching motion on MyoFullBody
uv run --extra c3d --extra smpl examples/retargeting/retarget_visualize.py \
  --c3d-file openbiomechanics/baseball_pitching/data/c3d/000002/000002_003034_73_207_002_FF_809.c3d

# Record a video
uv run --extra c3d --extra smpl examples/retargeting/retarget_visualize.py \
  --c3d-file path/to/file.c3d --record --output-dir ./recordings

# Use MyoBimanualArm model
uv run --extra c3d --extra smpl examples/retargeting/retarget_visualize.py \
  --c3d-file path/to/file.c3d --model MyoBimanualArm
```

### Via web viewer (Viser, browser-based)

```bash
uv run --extra c3d --extra smpl python -m musclemimic.web_viewer.run \
  --c3d-file openbiomechanics/baseball_pitching/data/c3d/000002/000002_003034_73_207_002_FF_809.c3d
```

### Pipeline steps

```
C3D file (Plug-in Gait markers, 360 Hz)
    │
    ▼ c3d_to_smpl.fit_smpl_to_c3d_cached()
    │  ├─ Label canonicalization (Plug-in Gait → MoSh++)
    │  ├─ Downsample to ~30 Hz
    │  ├─ Stage I: shape + marker coefficients (12 ref frames)
    │  └─ Stage II: per-frame pose + translation
    │
    ▼ AMASS-compatible dict {pose_aa, trans, betas, fps}
    │
    ▼ fit_smpl_motion()  [existing retargeting pipeline]
    │  └─ SMPL → MuJoCo IK → qpos/qvel trajectory
    │
    ▼ Visualization (MuJoCo renderer or Viser web viewer)
```

---

## 4. Trajectory Viewer

Visualizes retargeted AMASS motions on MyoFullBody / MyoBimanualArm with full muscle/tendon rendering.

```bash
# Single motion
uv run python -m musclemimic.web_viewer.run \
  --motion "KIT/6/WalkInCounterClockwiseCircle06_1_poses"

# Dataset group
uv run python -m musclemimic.web_viewer.run \
  --dataset-group AMASS_LOCOMOTION_DATASETS
```

---

## OpenBiomechanics C3D filename format

**Pitching:** `{USER}_{SESSION}_{HEIGHT}_{WEIGHT}_{PITCH#}_{TYPE}_{SPEED}.c3d`
- `000002_003034_73_207_002_FF_809` = 73in, 207lbs, pitch #2, Fastball, 80.9 mph

**Hitting:** `{USER}_{SESSION}_{HEIGHT}_{WEIGHT}_{SIDE}_{SWING#}_{EXITVELO}.c3d`
- `000004_000103_75_236_R_003_972` = 75in, 236lbs, Right-handed, swing #3, 97.2 mph exit velo
