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
import time
from pathlib import Path

from csd2smpl.data.synthesize import amass_sequence_files, synthesize_file


def _fmt_eta(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


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
    total = len(npz_files)
    print(f"Found {total} AMASS sequences under {args.amass_root}", flush=True)
    print(
        f"Marker layout: {args.layout}  placement: {args.placement}  "
        f"ssm: {args.ssm_json}",
        flush=True,
    )

    n_written = 0
    n_skipped = 0
    t_start = time.perf_counter()
    t_last_log = t_start
    for i, src in enumerate(npz_files):
        rel = src.relative_to(args.amass_root)
        dst = args.out_root / rel.with_suffix(".markers.npz")
        t_seq = time.perf_counter()
        wrote = synthesize_file(
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
        )
        if wrote:
            n_written += 1
        else:
            n_skipped += 1

        now = time.perf_counter()
        done = i + 1
        elapsed = now - t_start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else 0.0
        # Always log the first few, then throttle to once per 5 s.
        if done <= 3 or now - t_last_log >= 5.0 or done == total:
            print(
                f"  [{done:>4d}/{total}] wrote={n_written} skipped={n_skipped} "
                f"seq={now - t_seq:5.2f}s elapsed={_fmt_eta(elapsed)} "
                f"ETA={_fmt_eta(eta)}  {rel}",
                flush=True,
            )
            t_last_log = now

    total_elapsed = time.perf_counter() - t_start
    print(
        f"Done in {_fmt_eta(total_elapsed)}. Wrote {n_written}/{total} "
        f"(skipped {n_skipped}) to {args.out_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
