from __future__ import annotations

import time
from collections.abc import Sequence

import mujoco
import numpy as np
import trimesh.creation
import viser

from musclemimic.viewer.viser_utils import build_body_meshes


class TrajectoryViserViewer:
    _TENDON_RADIUS_SCALE = 0.35
    _TENDON_OPACITY_SCALE = 0.7

    def __init__(self, env, include_collision: bool = False, target_fps: float = 60.0) -> None:
        if getattr(env, "th", None) is None:
            raise ValueError("TrajectoryViserViewer requires an environment with env.th loaded.")
        if target_fps <= 0:
            raise ValueError(f"target_fps must be > 0, got {target_fps}.")

        self.env = env
        self.include_collision = include_collision
        self.target_fps = float(target_fps)

        self._server = None
        self._model = None
        self._data = None
        self._handles = {}
        self._tendon_handle = None
        self._tendon_scene = None
        self._tendon_option = None
        self._tendon_camera = None
        self._tendon_capacity = 0
        self._tendon_vertices = None
        self._tendon_faces = None

        self._motion_labels: list[str] = []
        self._current_traj = 0
        self._current_frame = 0
        self._paused = False
        self._loop = True
        self._time_multiplier = 1.0
        self._programmatic_update = False
        self._playback_remainder = 0.0

        self._pause_button = None
        self._traj_dropdown = None
        self._frame_slider = None
        self._loop_checkbox = None
        self._status_html = None

    def _setup_scene(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self._server = viser.ViserServer(label="trajectory-viewer")
        self._server.scene.configure_environment_map(environment_intensity=0.8)
        self._server.scene.add_grid(
            "/ground",
            width=10.0,
            height=10.0,
            width_segments=20,
            height_segments=20,
            plane="xy",
            cell_color=(180, 180, 180),
            section_color=(120, 120, 120),
            cell_thickness=1,
            section_thickness=2,
        )

        body_meshes = build_body_meshes(model, include_collision=self.include_collision)
        with self._server.atomic():
            for body_id, mesh in body_meshes.items():
                self._handles[body_id] = self._server.scene.add_batched_meshes_trimesh(
                    f"/bodies/{body_id}",
                    mesh,
                    batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=float),
                    batched_positions=np.array([[0.0, 0.0, 0.0]], dtype=float),
                    lod="off",
                    visible=True,
                )

        if model.ntendon > 0:
            maxgeom = max(10000, model.ngeom + model.ntendon * 20)
            self._tendon_scene = mujoco.MjvScene(model, maxgeom=maxgeom)
            self._tendon_option = mujoco.MjvOption()
            self._tendon_option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 1
            self._tendon_camera = mujoco.MjvCamera()
            unit_cylinder = trimesh.creation.cylinder(radius=1.0, height=2.0, sections=10)
            self._tendon_vertices = np.asarray(unit_cylinder.vertices, dtype=np.float32)
            self._tendon_faces = np.asarray(unit_cylinder.faces, dtype=np.uint32)
            positions, wxyzs, scales, colors, opacities = self._extract_tendon_mesh_instances(model, data)
            if positions.size > 0:
                self._create_tendon_handle(positions, wxyzs, scales, colors, opacities)

    def _setup_gui(self, motion_labels: Sequence[str]) -> None:
        self._motion_labels = list(motion_labels)
        tabs = self._server.gui.add_tab_group()

        with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
            with self._server.gui.add_folder("Playback"):
                self._pause_button = self._server.gui.add_button(
                    "Pause",
                    icon=viser.Icon.PLAYER_PAUSE,
                )

                @self._pause_button.on_click
                def _(_event) -> None:
                    self._paused = not self._paused
                    self._playback_remainder = 0.0
                    self._pause_button.label = "Play" if self._paused else "Pause"
                    self._pause_button.icon = (
                        viser.Icon.PLAYER_PLAY if self._paused else viser.Icon.PLAYER_PAUSE
                    )
                    self._update_status()

                speed_buttons = self._server.gui.add_button_group("Speed", options=["Slower", "Faster"])

                @speed_buttons.on_click
                def _(event) -> None:
                    if event.target.value == "Slower":
                        self._time_multiplier = max(0.1, self._time_multiplier / 2.0)
                    else:
                        self._time_multiplier = min(4.0, self._time_multiplier * 2.0)
                    self._update_status()

                restart_button = self._server.gui.add_button("Restart")

                @restart_button.on_click
                def _(_event) -> None:
                    self._playback_remainder = 0.0
                    self._render_frame(self._current_traj, 0)

            with self._server.gui.add_folder("Trajectory"):
                self._traj_dropdown = self._server.gui.add_dropdown(
                    "Motion",
                    options=tuple(self._motion_labels),
                    initial_value=self._motion_labels[0],
                )

                @self._traj_dropdown.on_update
                def _(event) -> None:
                    if self._programmatic_update:
                        return
                    new_traj = self._motion_labels.index(event.target.value)
                    self._playback_remainder = 0.0
                    self._render_frame(new_traj, 0)

                self._frame_slider = self._server.gui.add_slider(
                    "Frame",
                    min=0,
                    max=max(0, self.env.th.len_trajectory(0) - 1),
                    step=1,
                    initial_value=0,
                )

                @self._frame_slider.on_update
                def _(event) -> None:
                    if self._programmatic_update:
                        return
                    self._playback_remainder = 0.0
                    self._render_frame(self._current_traj, int(event.target.value))

                self._loop_checkbox = self._server.gui.add_checkbox("Loop", initial_value=self._loop)

                @self._loop_checkbox.on_update
                def _(event) -> None:
                    self._loop = bool(event.target.value)
                    self._update_status()

            self._status_html = self._server.gui.add_html("")

        self._update_status()

    def _render_frame(self, traj_idx: int, frame_idx: int) -> None:
        traj_len = self.env.th.len_trajectory(traj_idx)
        frame_idx = max(0, min(frame_idx, traj_len - 1))

        traj_data = self.env.th.traj.data.get(traj_idx, frame_idx, np)
        carry = self.env._additional_carry.replace(
            traj_state=self.env._additional_carry.traj_state.replace(
                traj_no=traj_idx,
                subtraj_step_no=frame_idx,
                subtraj_step_no_init=0,
            )
        )
        self.env._additional_carry = carry
        self._data = self.env.set_sim_state_from_traj_data(self._data, traj_data, carry)
        self.env._data = self._data
        mujoco.mj_forward(self._model, self._data)

        body_xpos = np.array(self._data.xpos)
        body_xquat = np.array(self._data.xquat)
        with self._server.atomic():
            for body_id, handle in self._handles.items():
                if body_id >= len(body_xpos):
                    continue
                handle.batched_positions = np.array([body_xpos[body_id]], dtype=float)
                handle.batched_wxyzs = np.array([body_xquat[body_id]], dtype=float)
            if self._tendon_handle is not None:
                self._update_tendon_handle(self._model, self._data)
        self._server.flush()

        self._current_traj = traj_idx
        self._current_frame = frame_idx

        self._programmatic_update = True
        try:
            self._traj_dropdown.value = self._motion_labels[traj_idx]
            self._frame_slider.max = max(0, traj_len - 1)
            self._frame_slider.value = frame_idx
            self._loop_checkbox.value = self._loop
        finally:
            self._programmatic_update = False

        self._update_status()

    def run(self, motion_labels: Sequence[str] | None = None) -> None:
        self.env.th.to_numpy()
        self.env.reset()

        cur = self.env
        while cur is not None and (not hasattr(cur, "model") or not hasattr(cur, "data")):
            cur = getattr(cur, "env", None)
        if cur is None:
            raise RuntimeError("Could not access MuJoCo model/data from environment.")

        self._model = cur.model
        self._data = cur.data

        if motion_labels is None or len(motion_labels) != self.env.th.n_trajectories:
            motion_labels = [f"Trajectory {i}" for i in range(self.env.th.n_trajectories)]

        self._setup_scene(self._model, self._data)
        self._setup_gui(motion_labels)
        self._render_frame(0, 0)

        target_dt = 1.0 / self.target_fps
        last_tick = time.perf_counter()

        try:
            while True:
                start = time.perf_counter()
                wall_dt = start - last_tick
                last_tick = start
                if not self._paused:
                    self._playback_remainder += wall_dt * self._time_multiplier
                    frames_to_advance = int(self._playback_remainder / self.env.th.traj_dt)

                    if frames_to_advance > 0:
                        self._playback_remainder -= frames_to_advance * self.env.th.traj_dt
                        traj_len = self.env.th.len_trajectory(self._current_traj)
                        next_frame = self._current_frame + frames_to_advance
                        if next_frame >= traj_len:
                            if self._loop:
                                next_frame %= traj_len
                            else:
                                next_frame = traj_len - 1
                                self._paused = True
                                self._playback_remainder = 0.0
                                self._pause_button.label = "Play"
                                self._pause_button.icon = viser.Icon.PLAYER_PLAY
                                self._update_status()
                        if next_frame != self._current_frame:
                            self._render_frame(self._current_traj, next_frame)

                elapsed = time.perf_counter() - start
                sleep_time = max(0.0, target_dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
        except KeyboardInterrupt:
            pass
        finally:
            if self._server is not None:
                self._server.stop()

    def _extract_tendon_mesh_instances(
        self, model: mujoco.MjModel, data: mujoco.MjData
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self._tendon_scene is None or self._tendon_option is None or self._tendon_camera is None:
            empty_vec3 = np.zeros((0, 3), dtype=np.float32)
            return empty_vec3, np.zeros((0, 4), dtype=np.float32), empty_vec3, np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.float32)

        mujoco.mjv_updateScene(
            model,
            data,
            self._tendon_option,
            None,
            self._tendon_camera,
            mujoco.mjtCatBit.mjCAT_ALL,
            self._tendon_scene,
        )

        ngeom = self._tendon_scene.ngeom
        if ngeom == 0:
            empty_vec3 = np.zeros((0, 3), dtype=np.float32)
            return empty_vec3, np.zeros((0, 4), dtype=np.float32), empty_vec3, np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.float32)

        geoms = self._tendon_scene.geoms[:ngeom]
        objtype_arr = np.array([geom.objtype for geom in geoms], dtype=np.int32)
        tendon_indices = np.where(objtype_arr == mujoco.mjtObj.mjOBJ_TENDON)[0]
        if len(tendon_indices) == 0:
            empty_vec3 = np.zeros((0, 3), dtype=np.float32)
            return empty_vec3, np.zeros((0, 4), dtype=np.float32), empty_vec3, np.zeros((0, 3), dtype=np.uint8), np.zeros((0,), dtype=np.float32)

        count = len(tendon_indices)
        positions = np.empty((count, 3), dtype=np.float32)
        wxyzs = np.empty((count, 4), dtype=np.float32)
        scales = np.empty((count, 3), dtype=np.float32)
        colors = np.empty((count, 3), dtype=np.uint8)
        opacities = np.empty((count,), dtype=np.float32)

        for i, geom_idx in enumerate(tendon_indices):
            geom = geoms[geom_idx]
            positions[i] = geom.pos
            quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, np.asarray(geom.mat, dtype=np.float64).reshape(-1))
            wxyzs[i] = quat.astype(np.float32)
            radius = max(1e-4, float(geom.size[0]) * self._TENDON_RADIUS_SCALE)
            scales[i] = np.array([radius, radius, geom.size[2]], dtype=np.float32)
            colors[i] = (np.clip(geom.rgba[:3], 0.0, 1.0) * 255.0).astype(np.uint8)
            opacities[i] = float(np.clip(geom.rgba[3], 0.0, 1.0)) * self._TENDON_OPACITY_SCALE

        return positions, wxyzs, scales, colors, opacities

    def _create_tendon_handle(
        self,
        positions: np.ndarray,
        wxyzs: np.ndarray,
        scales: np.ndarray,
        colors: np.ndarray,
        opacities: np.ndarray,
    ) -> None:
        if self._tendon_handle is not None:
            self._tendon_handle.remove()

        self._tendon_capacity = positions.shape[0]
        self._tendon_handle = self._server.scene.add_batched_meshes_simple(
            "/tendons",
            vertices=self._tendon_vertices,
            faces=self._tendon_faces,
            batched_positions=positions,
            batched_wxyzs=wxyzs,
            batched_scales=scales,
            batched_colors=colors,
            batched_opacities=opacities,
            lod="off",
            opacity=1.0,
            flat_shading=True,
            side="double",
            cast_shadow=False,
            receive_shadow=False,
            visible=True,
        )

    def _update_tendon_handle(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        positions, wxyzs, scales, colors, opacities = self._extract_tendon_mesh_instances(model, data)
        count = positions.shape[0]

        if count == 0:
            if self._tendon_handle is not None and self._tendon_capacity > 0:
                self._tendon_handle.batched_opacities = np.zeros((self._tendon_capacity,), dtype=np.float32)
            return

        if self._tendon_handle is None or count > self._tendon_capacity:
            self._create_tendon_handle(positions, wxyzs, scales, colors, opacities)
            return

        if count < self._tendon_capacity:
            pad = self._tendon_capacity - count
            positions = np.concatenate([positions, np.zeros((pad, 3), dtype=np.float32)], axis=0)
            wxyzs = np.concatenate([wxyzs, np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (pad, 1))], axis=0)
            scales = np.concatenate([scales, np.zeros((pad, 3), dtype=np.float32)], axis=0)
            colors = np.concatenate([colors, np.zeros((pad, 3), dtype=np.uint8)], axis=0)
            opacities = np.concatenate([opacities, np.zeros((pad,), dtype=np.float32)], axis=0)

        self._tendon_handle.batched_positions = positions
        self._tendon_handle.batched_wxyzs = wxyzs
        self._tendon_handle.batched_scales = scales
        self._tendon_handle.batched_colors = colors
        self._tendon_handle.batched_opacities = opacities

    def _update_status(self) -> None:
        if self._status_html is None:
            return

        traj_len = self.env.th.len_trajectory(self._current_traj)
        time_s = self._current_frame * self.env.th.traj_dt
        label = self._motion_labels[self._current_traj] if self._motion_labels else str(self._current_traj)
        self._status_html.content = (
            f"<b>Motion:</b> {label}"
            f" &nbsp; <b>Frame:</b> {self._current_frame + 1}/{traj_len}"
            f" &nbsp; <b>Time:</b> {time_s:.2f}s"
            f" &nbsp; <b>Speed:</b> {self._time_multiplier:.2f}x"
            f" &nbsp; <b>Paused:</b> {self._paused}"
        )
