# External assets

Files here are **fetched at setup time**, not vendored. The upstream assets
are released by Max Planck (MPG) under a license that allows
non-commercial research use but forbids redistribution without written
permission, so this repo does not redistribute them.

Run:

    bash csd2smpl/scripts/fetch_external.sh

to populate this directory.

## Contents (after fetch)

| File | Source | Use |
|---|---|---|
| `ssm_all_marker_placements.json` | [nghorbani/amass](https://github.com/nghorbani/amass/blob/master/src/amass/data/ssm_all_marker_placements.json) | Per-subject SSM marker → SMPL-mesh vertex indices. Drives the vertex-based marker placement in synthesis (v2, more accurate than the joint-anchored v1 fallback). |

## Citation

AMASS: Archive of Motion Capture as Surface Shapes.
Mahmood, Ghorbani, Troje, Pons-Moll, Black. ICCV 2019.
https://amass.is.tue.mpg.de
