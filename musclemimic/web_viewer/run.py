from __future__ import annotations

import argparse

from loco_mujoco.task_factories import AMASSDatasetConf, CustomDatasetConf, ImitationFactory

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
        "--c3d-file",
        type=str,
        default=None,
        help="Path to a C3D file. Fits SMPL, retargets, and visualizes on the musculoskeletal model.",
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


def _make_env_from_c3d(args) -> tuple:
    """Fit SMPL to C3D markers, retarget, and create environment."""
    import logging
    import os
    from pathlib import Path

    from loco_mujoco.smpl.retargeting import (
        OPTIMIZED_SHAPE_FILE_NAME,
        fit_gmr_motion,
        fit_smpl_motion,
        fit_smpl_shape,
        get_converted_amass_dataset_path,
        get_smpl_model_path,
        load_robot_conf_file,
    )

    from .c3d_to_smpl import fit_smpl_to_c3d_cached, save_motion_data_as_amass_smplh_npz

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("c3d_pipeline")

    c3d_path = args.c3d_file
    if not Path(c3d_path).exists():
        raise SystemExit(f"C3D file not found: {c3d_path}")

    smpl_model_path = get_smpl_model_path()
    if not Path(smpl_model_path).exists():
        raise SystemExit(
            f"SMPL model not found at {smpl_model_path}. "
            "Run: musclemimic-set-smpl-model-path <path>"
        )

    # Step 1: C3D → SMPL parameters
    log.info(f"[1/3] Fitting SMPL to C3D: {c3d_path}")
    motion_data = fit_smpl_to_c3d_cached(c3d_path, smpl_model_path)

    # Step 2: SMPL → retargeted MuJoCo trajectory
    log.info("[2/3] Retargeting SMPL → musculoskeletal model...")
    robot_conf = load_robot_conf_file(args.model)
    if args.retargeting_method == "gmr":
        gmr_motion_path = save_motion_data_as_amass_smplh_npz(
            motion_data,
            Path(c3d_path).parent / ".smpl_cache" / f"{Path(c3d_path).stem}_smplh_for_gmr.npz",
        )
        trajectory, analysis = fit_gmr_motion(
            args.model,
            robot_conf,
            str(gmr_motion_path),
            log,
            {
                "src_human": args.gmr_src_human,
                "target_fps": args.gmr_target_fps,
                "solver": args.gmr_solver,
                "damping": args.gmr_damping,
                "offset_to_ground": args.gmr_offset_to_ground,
                "use_velocity_limit": args.gmr_use_velocity_limit,
            },
        )
    else:
        trajectory, analysis = fit_smpl_motion(
            args.model,
            robot_conf,
            smpl_model_path,
            motion_data,
            path_to_optimized_smpl_shape=_get_or_fit_shape(args.model, robot_conf, smpl_model_path, log),
            logger=log,
        )

    # Step 3: Create environment with the trajectory
    log.info("[3/3] Loading into viewer...")
    env = ImitationFactory.make(
        args.model,
        custom_dataset_conf=CustomDatasetConf(traj=trajectory),
        env_params={"timestep": 0.002, "n_substeps": 5},
    )

    label = Path(c3d_path).stem
    return env, [label]
def _get_or_fit_shape(env_name, robot_conf, smpl_model_path, log):
    """Get cached optimized SMPL shape, or fit a new one."""
    import os

    from loco_mujoco.smpl.retargeting import (
        OPTIMIZED_SHAPE_FILE_NAME,
        fit_smpl_shape,
        get_converted_amass_dataset_path,
    )

    cache_env = env_name.replace("Mjx", "") if "Mjx" in env_name else env_name
    path = os.path.join(get_converted_amass_dataset_path(), cache_env, OPTIMIZED_SHAPE_FILE_NAME)

    if not os.path.exists(path):
        log.info("Fitting SMPL shape to robot (one-time)...")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fit_smpl_shape(env_name, robot_conf, smpl_model_path, path, log)

    return path


def main() -> None:
    args = parse_args()

    if args.c3d_file:
        env, motion_labels = _make_env_from_c3d(args)
    else:
        motions = args.motion or []
        if bool(motions) == bool(args.dataset_group):
            raise SystemExit("Pass exactly one of --motion, --dataset-group, or --c3d-file.")

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
