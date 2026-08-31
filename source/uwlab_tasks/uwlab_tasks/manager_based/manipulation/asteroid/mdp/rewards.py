# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for pick-only tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg

from uwlab_tasks.manager_based.manipulation.omnireset.assembly_keypoints import Offset
from uwlab_tasks.manager_based.manipulation.omnireset.mdp import utils
from uwlab_tasks.manager_based.manipulation.omnireset.mdp.success_monitor_cfg import SuccessMonitorCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ProgressContextPickOnly(ManagerTermBase):
    """Pick-only success context (no receptive object).

    Success is ``insertive object lifted above pick_height_threshold`` AND ``gripper pointing
    vertically down``. Other reward / termination terms read :attr:`success`,
    :attr:`insertive_asset_z` and :attr:`continuous_success_counter` from this term.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.params.get("insertive_asset_cfg").name]  # type: ignore

        insertive_meta = utils.read_metadata_from_usd_directory(self.insertive_asset.cfg.spawn.usd_path)
        self.insertive_asset_offset = Offset(
            pos=tuple(insertive_meta.get("assembled_offset").get("pos")),
            quat=tuple(insertive_meta.get("assembled_offset").get("quat")),
        )

        # Gripper orientation tracking: success additionally requires the gripper to point
        # vertically down. The gripper's approach axis is the local axis given by
        # ``gripper_approach_direction`` in the robot metadata (local +x for the Robotiq 2f85);
        # "pointing down" means that axis, expressed in world frame, aligns with world -z.
        self.robot: Articulation = env.scene[cfg.params.get("robot_asset_cfg").name]  # type: ignore
        robot_meta = utils.read_metadata_from_usd_directory(self.robot.cfg.spawn.usd_path)
        approach_dir = robot_meta.get("gripper_approach_direction", [1.0, 0.0, 0.0])
        self.gripper_approach_dir = torch.tensor(approach_dir, dtype=torch.float32, device=env.device).view(1, 3)
        self.gripper_body_id = self.robot.find_bodies(cfg.params.get("robot_asset_cfg").body_names)[0][0]
        self.gripper_pointing_down = torch.zeros((env.num_envs), dtype=torch.bool, device=env.device)

        self.insertive_asset_z = torch.zeros((env.num_envs), device=env.device)
        self.success = torch.zeros((self._env.num_envs), dtype=torch.bool, device=self._env.device)
        self.continuous_success_counter = torch.zeros((self._env.num_envs), dtype=torch.int32, device=self._env.device)

        success_monitor_cfg = SuccessMonitorCfg(monitored_history_len=100, num_monitored_data=1, device=env.device)
        self.success_monitor = success_monitor_cfg.class_type(success_monitor_cfg)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        super().reset(env_ids)
        self.continuous_success_counter[:] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        insertive_asset_cfg: SceneEntityCfg,
        robot_asset_cfg: SceneEntityCfg,
        command_context: str = "task_command",
        pick_height_threshold: float = 0.02,
        gripper_down_dot_threshold: float = 0.9,
    ) -> torch.Tensor:
        # Object lifted above threshold?
        insertive_z_pos = self.insertive_asset.data.root_pos_w[:, 2]
        self.insertive_asset_z[:] = insertive_z_pos

        # Gripper pointing vertically down? Rotate the local approach axis into world frame and
        # require its z-component to be close to -1. A threshold of -1.0 disables the constraint.
        if gripper_down_dot_threshold <= -1.0:
            self.gripper_pointing_down[:] = True
        else:
            gripper_quat_w = self.robot.data.body_link_quat_w[:, self.gripper_body_id]
            approach_w = math_utils.quat_apply(gripper_quat_w, self.gripper_approach_dir.expand(env.num_envs, 3))
            self.gripper_pointing_down[:] = (-approach_w[:, 2]) >= gripper_down_dot_threshold

        self.success[:] = (insertive_z_pos > pick_height_threshold) & self.gripper_pointing_down

        self.continuous_success_counter[:] = torch.where(
            self.success, self.continuous_success_counter + 1, torch.zeros_like(self.continuous_success_counter)
        )
        self.success_monitor.success_update(
            torch.zeros(env.num_envs, dtype=torch.int32, device=env.device), self.success
        )

        return torch.zeros(env.num_envs, device=env.device)


def dense_success_reward_pick_only(
    env: ManagerBasedRLEnv, std: float, context: str = "progress_context"
) -> torch.Tensor:
    """Dense shaping toward lifting the object: ``exp(-max(0, 0.4 - z) / std)``."""
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    insertive_asset_z: torch.Tensor = getattr(context_term, "insertive_asset_z")
    return torch.exp(-torch.clamp(0.4 - insertive_asset_z, min=0) / std)


def success_reward_pick_only(env: ManagerBasedRLEnv, context: str = "progress_context") -> torch.Tensor:
    """Sparse reward: 1 while :class:`ProgressContextPickOnly` reports success."""
    context_term: ManagerTermBase = env.reward_manager.get_term_cfg(context).func  # type: ignore
    success: torch.Tensor = getattr(context_term, "success")
    return torch.where(success, 1.0, 0.0)
