# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event terms for pick-only tasks and ASTEROID domain randomization."""

from __future__ import annotations

import os
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.actions.task_space_actions import RelCartesianOSCAction
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.events import MultiResetManager, sample_state_data_set
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class SingleObjectMultiResetManager(MultiResetManager):
    """:class:`~uwlab_tasks.manager_based.manipulation.omnireset.mdp.events.MultiResetManager`
    for scenes with only an insertive object.

    OmniReset keys reset datasets by the object *pair* directory (``Peg__PegHole``); pick-only
    scenes have no receptive object, so datasets are keyed by the insertive object alone
    (``<dataset_dir>/Resets/<Object>/resets_<ResetType>.pt``). Sampling and state restoration
    are inherited unchanged.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        # NOTE: deliberately skips MultiResetManager.__init__, which requires a receptive object.
        ManagerTermBase.__init__(self, cfg, env)

        dataset_dir: str = cfg.params.get("dataset_dir", "")
        reset_types: list[str] = cfg.params.get("reset_types", [])
        probabilities: list[float] = cfg.params.get("probs", [])

        if not reset_types:
            raise ValueError("No reset_types provided")
        if len(reset_types) != len(probabilities):
            raise ValueError("Number of reset_types must match number of probabilities")

        insertive_usd_path = env.scene["insertive_object"].cfg.spawn.usd_path
        pair = utils.object_name_from_usd(insertive_usd_path)

        dataset_files = [f"{dataset_dir}/Resets/{pair}/resets_{rt}.pt" for rt in reset_types]

        self.datasets = []
        num_states = []
        for dataset_file in dataset_files:
            local_file_path = utils.safe_retrieve_file_path(dataset_file)
            if not os.path.exists(local_file_path):
                raise FileNotFoundError(f"Dataset file {dataset_file} could not be accessed or downloaded.")

            dataset = torch.load(local_file_path)
            num_states.append(len(dataset["initial_state"]["articulation"]["robot"]["joint_position"]))
            init_indices = torch.arange(num_states[-1], device=env.device)
            self.datasets.append(sample_state_data_set(dataset, init_indices, env.device))

        self.probs = torch.tensor(probabilities, device=env.device) / sum(probabilities)
        self.num_states = torch.tensor(num_states, device=env.device)
        self.num_tasks = len(self.datasets)

        if cfg.params.get("success") is not None:
            success_monitor_cfg = SuccessMonitorCfg(
                monitored_history_len=100, num_monitored_data=self.num_tasks, device=env.device
            )
            self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

        self.task_id = torch.randint(0, self.num_tasks, (self.num_envs,), device=self.device)


class reset_root_states_discrete_grid(ManagerTermBase):
    """Reset root states by sampling x/y around discrete grid centers.

    X/Y centers are generated from the configured pose range using ``grid_shape``. Each reset
    samples one center per env, applies small uniform x/y jitter, then samples the remaining
    pose dimensions uniformly from ``pose_range``.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        pose_range_dict = cfg.params.get("pose_range")
        velocity_range_dict = cfg.params.get("velocity_range")

        self.pose_range = torch.tensor(
            [pose_range_dict.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=env.device,
        )
        self.velocity_range = torch.tensor(
            [velocity_range_dict.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]],
            device=env.device,
        )
        self.asset_cfgs = list(cfg.params.get("asset_cfgs", dict()).values())
        self.offset_asset_cfg = cfg.params.get("offset_asset_cfg")
        self.use_bottom_offset = cfg.params.get("use_bottom_offset", False)

        grid_shape = cfg.params.get("grid_shape", (3, 3))
        if len(grid_shape) != 2:
            raise ValueError("grid_shape must be a 2-tuple, e.g. (3, 3)")
        num_x, num_y = grid_shape
        x_centers = torch.linspace(self.pose_range[0, 0], self.pose_range[0, 1], num_x, device=env.device)
        y_centers = torch.linspace(self.pose_range[1, 0], self.pose_range[1, 1], num_y, device=env.device)
        grid_x, grid_y = torch.meshgrid(x_centers, y_centers, indexing="ij")
        self.xy_grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

        self.xy_noise_range = torch.tensor(cfg.params.get("xy_noise_range", (-0.01, 0.01)), device=env.device)

        if self.use_bottom_offset:
            self.bottom_offset_positions = dict()
            for asset_cfg in self.asset_cfgs:
                asset: RigidObject | Articulation = env.scene[asset_cfg.name]
                metadata = utils.read_metadata_from_usd_directory(asset.cfg.spawn.usd_path)
                bottom_offset = metadata.get("bottom_offset")
                self.bottom_offset_positions[asset_cfg.name] = (
                    torch.tensor(bottom_offset.get("pos"), device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
                )
                assert tuple(bottom_offset.get("quat")) == (1.0, 0.0, 0.0, 0.0), (
                    "Bottom offset rotation must be (1.0, 0.0, 0.0, 0.0)"
                )

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfgs: dict[str, SceneEntityCfg] = dict(),
        offset_asset_cfg: SceneEntityCfg = None,
        use_bottom_offset: bool = False,
        grid_shape: tuple[int, int] = (3, 3),
        xy_noise_range: tuple[float, float] = (-0.01, 0.01),
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)

        num_envs = len(env_ids)
        rand_pose_samples = math_utils.sample_uniform(
            self.pose_range[:, 0], self.pose_range[:, 1], (num_envs, 6), device=env.device
        )

        grid_ids = torch.randint(0, self.xy_grid.shape[0], (num_envs,), device=env.device)
        xy = self.xy_grid[grid_ids]
        xy_noise = math_utils.sample_uniform(
            self.xy_noise_range[0], self.xy_noise_range[1], (num_envs, 2), device=env.device
        )
        rand_pose_samples[:, 0:2] = (xy + xy_noise).clamp(min=self.pose_range[0:2, 0], max=self.pose_range[0:2, 1])

        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_pose_samples[:, 3], rand_pose_samples[:, 4], rand_pose_samples[:, 5]
        )
        rand_vel_samples = math_utils.sample_uniform(
            self.velocity_range[:, 0], self.velocity_range[:, 1], (num_envs, 6), device=env.device
        )

        for asset_cfg in self.asset_cfgs:
            asset: RigidObject | Articulation = env.scene[asset_cfg.name]
            root_states = asset.data.default_root_state[env_ids].clone()
            positions = root_states[:, 0:3] + env.scene.env_origins[env_ids] + rand_pose_samples[:, 0:3]

            if self.offset_asset_cfg:
                offset_asset: RigidObject | Articulation = env.scene[self.offset_asset_cfg.name]
                offset_positions = offset_asset.data.default_root_state[env_ids].clone()
                positions += offset_positions[:, 0:3]

            if self.use_bottom_offset:
                positions -= self.bottom_offset_positions[asset_cfg.name][env_ids, 0:3]

            orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
            velocities = root_states[:, 7:13] + rand_vel_samples

            asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
            asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)


class randomize_env_cfg_unified(ManagerTermBase):
    """Coupled domain randomization over arm joint dynamics, OSC gains and action scaling.

    Samples one ``coupled_progress`` scalar per env and maps it to arm sysid (armature /
    friction), actuator delay and :class:`RelCartesianOSCAction` Kp/Kd, so the environment
    always lies in a tractable region (high joint friction with soft gains is unsolvable).
    Action scaling is unrelated to task feasibility and is randomized independently.

    * arm sysid range: ``[0, full sysid-randomized values]``
    * delay range: ``[delay_min, delay_max]``
    * OSC controller range: ``[action cfg defaults, terminal_kp / terminal_damping_ratio]``
    * action scaling range: ``[initial_scales, target_scales]``
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.robot: Articulation = env.scene[self.asset_cfg.name]
        self.joint_ids = self.robot.find_joints(cfg.params["joint_names"])[0]
        self.actuator_name: str = cfg.params["actuator_name"]
        self._action_name: str = cfg.params["action_name"]
        self._action_term: RelCartesianOSCAction | None = None

        metadata = utils.read_metadata_from_usd_directory(self.robot.cfg.spawn.usd_path)
        sysid = metadata["sysid"]
        self.armature = sysid["armature"]
        self.static_friction = sysid["static_friction"]
        self.dynamic_ratio = sysid["dynamic_ratio"]
        self.viscous_friction = sysid["viscous_friction"]

    def _resolve_action_term(self):
        if self._action_term is not None:
            return
        action_term = self._env.action_manager._terms.get(self._action_name)
        if action_term is None or not isinstance(action_term, RelCartesianOSCAction):
            raise ValueError(f"Action term '{self._action_name}' is not a RelCartesianOSCAction.")
        # The action term stores a single (6,) scale; promote it to per-env (num_envs, 6) so we can
        # randomize it per env. ``process_actions`` broadcasts either layout.
        if action_term._scale.dim() == 1:
            action_term._scale = action_term._scale.unsqueeze(0).expand(self.num_envs, -1).clone()
        self._action_term = action_term

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        asset_cfg: SceneEntityCfg,
        joint_names: list[str],
        actuator_name: str,
        action_name: str,
        arm_scale_range: tuple[float, float] = (0.8, 1.2),
        delay_range: tuple[int, int] = (0, 1),
        kp_scale_range: tuple[float, float] = (0.8, 1.2),
        terminal_kp: tuple[float, ...] = (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
        terminal_damping_ratio: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        initial_scales: tuple[float, ...] = (0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
        target_scales: tuple[float, ...] = (0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
        coupled_progress_range: tuple[float, float] = (0.0, 1.0),
        action_scale_progress_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._resolve_action_term()

        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=self.robot.device)

        n = len(env_ids)
        n_joints = len(self.joint_ids)
        device = self.robot.device

        c_lo, c_hi = coupled_progress_range
        coupled_progress = c_lo + torch.rand(n, 1, device=device) * (c_hi - c_lo)

        def _scale_sysid(nominal, scale_range):
            lo, hi = scale_range
            val = torch.as_tensor(nominal, device=device, dtype=torch.float32)
            return val * (lo + torch.rand(n, n_joints, device=device) * (hi - lo))

        arm_vals = _scale_sysid(self.armature, arm_scale_range) * coupled_progress
        sfric_vals = _scale_sysid(self.static_friction, arm_scale_range) * coupled_progress
        dratio_vals = _scale_sysid(self.dynamic_ratio, arm_scale_range) * coupled_progress
        dfric_vals = torch.minimum(dratio_vals * sfric_vals, sfric_vals)
        vfric_vals = _scale_sysid(self.viscous_friction, arm_scale_range) * coupled_progress

        self.robot.write_joint_armature_to_sim(arm_vals, joint_ids=self.joint_ids, env_ids=env_ids)
        self.robot.write_joint_friction_coefficient_to_sim(
            sfric_vals,
            joint_dynamic_friction_coeff=dfric_vals,
            joint_viscous_friction_coeff=vfric_vals,
            joint_ids=self.joint_ids,
            env_ids=env_ids,
        )

        delay_lo, delay_hi = delay_range
        if delay_hi > delay_lo:
            actuator = self.robot.actuators[self.actuator_name]
            if hasattr(actuator, "positions_delay_buffer"):
                max_delay = delay_lo + torch.round(coupled_progress.squeeze(-1) * float(delay_hi - delay_lo)).to(
                    dtype=torch.int32
                )
                min_delay = torch.full_like(max_delay, fill_value=delay_lo)
                span = (max_delay - min_delay + 1).clamp(min=1)
                # Vectorized integer sampling with per-env bounds in [min_delay, max_delay].
                delays = min_delay + torch.floor(torch.rand(n, device=device) * span.to(torch.float32)).to(torch.int32)
                actuator.positions_delay_buffer.set_time_lag(delays, env_ids)
                actuator.velocities_delay_buffer.set_time_lag(delays, env_ids)
                actuator.efforts_delay_buffer.set_time_lag(delays, env_ids)

        k_lo, k_hi = kp_scale_range
        s_xyz = k_lo + torch.rand(n, 1, device=device) * (k_hi - k_lo)
        s_rpy = k_lo + torch.rand(n, 1, device=device) * (k_hi - k_lo)
        s_dr_xyz = k_lo + torch.rand(n, 1, device=device) * (k_hi - k_lo)
        s_dr_rpy = k_lo + torch.rand(n, 1, device=device) * (k_hi - k_lo)

        kp_default = self._action_term._kp_default
        dr_default = self._action_term._damping_ratio_default
        kp_term = torch.tensor(terminal_kp, device=device, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
        dr_term = torch.tensor(terminal_damping_ratio, device=device, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
        kp_term[:, :3] *= s_xyz
        kp_term[:, 3:] *= s_rpy
        dr_term[:, :3] *= s_dr_xyz
        dr_term[:, 3:] *= s_dr_rpy

        new_kp = kp_default.unsqueeze(0) + coupled_progress * (kp_term - kp_default.unsqueeze(0))
        new_dr = dr_default.unsqueeze(0) + coupled_progress * (dr_term - dr_default.unsqueeze(0))
        self._action_term._kp[env_ids] = new_kp
        self._action_term._kd[env_ids] = 2.0 * torch.sqrt(new_kp) * new_dr

        a_lo, a_hi = action_scale_progress_range
        action_progress = a_lo + torch.rand(n, 1, device=device) * (a_hi - a_lo)
        initial = torch.tensor(initial_scales, device=device, dtype=torch.float32).unsqueeze(0)
        target = torch.tensor(target_scales, device=device, dtype=torch.float32).unsqueeze(0)
        self._action_term._scale[env_ids] = initial + action_progress * (target - initial)


class randomize_gripper_pos_affine(ManagerTermBase):
    """Per-env affine noise on the ``gripper_pos`` observation: ``pos -> pos * scale + offset``.

    Stores ``(num_envs,)`` ``scale`` and ``offset`` tensors that
    :func:`~.observations.gripper_pos_normalized` reads via
    ``env.event_manager.get_term_cfg(<term_name>).func``. Both are resampled per env at each
    reset and fixed within an episode.

    Makes the policy robust to calibration / encoder drift on the real Robotiq's POS register,
    so it cannot latch onto an absolute threshold like ``pos > 0.93`` to detect a grasp.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.scale = torch.ones(env.num_envs, device=env.device)
        self.offset = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids,
        scale_range: tuple[float, float] = (0.75, 1.25),
        offset_range: tuple[float, float] = (-0.1, 0.1),
    ) -> None:
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        n = env_ids.shape[0]
        s_lo, s_hi = scale_range
        o_lo, o_hi = offset_range
        self.scale[env_ids] = s_lo + torch.rand(n, device=env.device) * (s_hi - s_lo)
        self.offset[env_ids] = o_lo + torch.rand(n, device=env.device) * (o_hi - o_lo)
