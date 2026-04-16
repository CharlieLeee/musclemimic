"""Stage-1 entry point: synthesize a C3D-style dataset from AMASS.

Usage
-----
::

    python -m csd2smpl.scripts.synthesize_dataset \\
        --amass_root /data/amass \\
        --out_root   /data/amass_c3d \\
        --model_path body_models/smpl \\
        --noise_std  0.01 \\
        --dropout_p  0.05 \\
        --target_fps 30
"""

from __future__ import annotations

import argparse
from pathlib import Path

from csd2smpl.data.synthesize import amass_sequence_files, synthesize_file


def main() -> None:
    """Walk an AMASS tree and write a C3D NPZ next to each input."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--amass_root", required=True, type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--model_path", default="body_models/smpl")
    parser.add_argument("--layout", default="cmu_41")
    parser.add_argument(
        "--placement", default="auto", choices=["vertex", "joint_offset", "auto"],
        help="marker attachment: vertex uses SSM mesh IDs, joint_offset uses "
        "joint-anchored approximations, auto picks vertex if ssm_json is present",
    )
    parser.add_argument(
        "--ssm_json", type=Path,
        default=Path(__file__).parents[1] / "data" / "external" / "ssm_all_marker_placements.json",
        help="path to SSM marker→vertex JSON (from scripts/fetch_external.sh)",
    )
    parser.add_argument("--noise_std", type=float, default=0.01)
    parser.add_argument("--dropout_p", type=float, default=0.05)
    parser.add_argument("--target_fps", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    npz_files = amass_sequence_files(args.amass_root)
    print(f"Found {len(npz_files)} AMASS sequences under {args.amass_root}")
    print(f"Marker layout: {args.layout}  placement: {args.placement}  ssm: {args.ssm_json}")

    n_written = 0
    for i, src in enumerate(npz_files):
        rel = src.relative_to(args.amass_root)
        dst = args.out_root / rel.with_suffix(".markers.npz")
        if synthesize_file(
            npz_path=src,
            out_path=dst,
            model_path=args.model_path,
            layout_name=args.layout,
            placement=args.placement,
            ssm_json_path=args.ssm_json,
            noise_std=args.noise_std,
            dropout_p=args.dropout_p,
            target_fps=args.target_fps,
            device=args.device,
        ):
            n_written += 1
        if i % 500 == 0:
            print(f"  {i}/{len(npz_files)}: {rel}")

    print(f"Done. Wrote {n_written}/{len(npz_files)} files to {args.out_root}")


if __name__ == "__main__":
    main()
