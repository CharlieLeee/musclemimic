# Web Viewer

Browser-based visualization tools built on [Viser](https://github.com/nerfstudio-project/viser).

## C3D Motion Capture Viewer

Interactive 3D viewer for C3D biomechanics files (e.g. [OpenBiomechanics](https://github.com/drivelineresearch/openbiomechanics)).

### Setup

Install the `c3d` optional dependency:

```bash
uv sync --extra c3d
```

### Get sample data

Sparse-checkout one subject (~5 files) instead of the full 2.2 GB repo:

```bash
git clone --filter=blob:none --sparse https://github.com/drivelineresearch/openbiomechanics.git
cd openbiomechanics

# Pitching data
git sparse-checkout set baseball_pitching/data/c3d/000002

# Hitting data (includes bat markers)
git sparse-checkout add baseball_hitting/data/c3d/000004
```

### Usage

```bash
# Basic
uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer path/to/file.c3d

# With marker trails and custom port
uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer path/to/file.c3d --trail 10 --port 9090
```

Then open http://localhost:8080 in your browser.

### Examples

```bash
# Pitching trial (80.9 mph fastball)
uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer \
  openbiomechanics/baseball_pitching/data/c3d/000002/000002_003034_73_207_002_FF_809.c3d

# Hitting trial (97.2 mph exit velo)
uv run --extra c3d python -m musclemimic.web_viewer.c3d_viewer \
  openbiomechanics/baseball_hitting/data/c3d/000004/000004_000103_75_236_R_003_972.c3d --trail 15
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8080 | Viser server port |
| `--trail` | 0 | Trail length in frames (0 = off) |
| `--marker-size` | 0.012 | Marker radius in meters |

### Viewer controls

- **Play/Pause** -- start/stop playback
- **Speed** -- 0.25x / 0.5x / 1x / 2x
- **Frame slider** -- scrub to any frame
- **Loop** -- toggle looping
- **Marker size** -- adjust marker radius
- **Trail length** -- show fading marker history
- **Show skeleton / Show markers** -- toggle visibility

### Color convention

- Blue: left-side markers/bones
- Red: right-side markers/bones
- Gray: center markers (head, torso, pelvis)
- Gold: bat markers (hitting files)

### Supported data

Works with any C3D file using the Plug-in Gait marker set. Automatically detects:
- Body markers (45 standard markers)
- Bat markers (`Marker1`-`Marker10`, connected sequentially)
- Units (meters or millimeters)

### C3D filename format (OpenBiomechanics)

**Pitching:** `{USER}_{SESSION}_{HEIGHT}_{WEIGHT}_{PITCH#}_{TYPE}_{SPEED}.c3d`
- e.g. `000002_003034_73_207_002_FF_809` = 73in, 207lbs, pitch #2, Fastball, 80.9 mph

**Hitting:** `{USER}_{SESSION}_{HEIGHT}_{WEIGHT}_{SIDE}_{SWING#}_{EXITVELO}.c3d`
- e.g. `000004_000103_75_236_R_003_972` = 75in, 236lbs, Right-handed, swing #3, 97.2 mph exit velo

## Trajectory Viewer

Visualizes retargeted AMASS motions on MyoFullBody / MyoBimanualArm models. See `run.py` for usage.
