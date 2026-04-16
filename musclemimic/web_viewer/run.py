from __future__ import annotations

import argparse

from loco_mujoco.task_factories import AMASSDatasetConf, ImitationFactory

from .trajectory_viewer import TrajectoryViserViewer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Web-based trajectory viewer for retargeted MuscleMimic motions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["MyoBimanualArm", "MyoFullBody"],
        default="MyoFullBody",
        help="Model type to visualize.",
    )
    parser.add_argument(
        "--motion",
        action="append",
        default=None,
        help="Motion name (repeatable).",
    )
    parser.add_argument(
        "--dataset-group",
        type=str,
        default=None,
        help="Dataset group from loco_mujoco.smpl.const.",
    )
    parser.add_argument(
        "--retargeting-method",
        choices=["smpl", "gmr"],
        default="smpl",
        help="Retargeting method.",
    )
    parser.add_argument("--gmr-src-human", default="smplh", help="Source human model for GMR.")
    parser.add_argument("--gmr-target-fps", type=int, default=30, help="Target FPS for GMR retargeting.")
    parser.add_argument("--gmr-solver", default="daqp", help="IK solver for GMR.")
    parser.add_argument("--gmr-damping", type=float, default=0.5, help="Damping factor for GMR.")
    parser.add_argument(
        "--gmr-offset-to-ground",
        action="store_true",
        default=False,
        help="Offset the trajectory to the ground plane.",
    )
    parser.add_argument(
        "--gmr-use-velocity-limit",
        action="store_true",
        default=False,
        help="Use GMR velocity limits.",
    )
    parser.add_argument(
        "--include-collision",
        action="store_true",
        default=False,
        help="Render collision geoms instead of visual geoms.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    motions = args.motion or []
    if bool(motions) == bool(args.dataset_group):
        raise SystemExit("Pass exactly one of --motion or --dataset-group.")

    if motions:
        dataset_conf = AMASSDatasetConf(motions)
    else:
        dataset_conf = AMASSDatasetConf(dataset_group=args.dataset_group)

    dataset_conf.retargeting_method = args.retargeting_method
    if args.retargeting_method == "gmr":
        dataset_conf.gmr_config = {
            "src_human": args.gmr_src_human,
            "target_fps": args.gmr_target_fps,
            "solver": args.gmr_solver,
            "damping": args.gmr_damping,
            "offset_to_ground": args.gmr_offset_to_ground,
            "use_velocity_limit": args.gmr_use_velocity_limit,
        }

    env = ImitationFactory.make(
        args.model,
        amass_dataset_conf=dataset_conf,
        env_params={"timestep": 0.002, "n_substeps": 5},
    )

    motion_labels = list(dict.fromkeys(motions)) if motions else [f"Trajectory {i}" for i in range(env.th.n_trajectories)]
    TrajectoryViserViewer(env, include_collision=args.include_collision).run(motion_labels)


if __name__ == "__main__":
    main()
