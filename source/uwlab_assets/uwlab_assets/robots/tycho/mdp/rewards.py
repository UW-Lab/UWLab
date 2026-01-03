# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def punish_hand_tilted(
    env: ManagerBasedRLEnv,
    fixed_chop_frame_cfg: SceneEntityCfg = SceneEntityCfg("frame_fixed_chop_tip"),
    free_chop_frame_cfg: SceneEntityCfg = SceneEntityCfg("frame_free_chop_tip"),
):
    fixed_chop_frame: FrameTransformer = env.scene[fixed_chop_frame_cfg.name]
    free_chop_frame: FrameTransformer = env.scene[free_chop_frame_cfg.name]

    ee_tip_fixed_height_w = fixed_chop_frame.data.target_pos_w[..., 0, 2]
    ee_tip_free_height_w = free_chop_frame.data.target_pos_w[..., 0, 2]

    height_diff = -torch.abs(ee_tip_fixed_height_w - ee_tip_free_height_w)
    height_diff_punish = torch.clamp(height_diff * 50, -1)

    return height_diff_punish


def punish_touching_ground(env: ManagerBasedRLEnv, robot_str: str = "robot") -> torch.Tensor:
    ee_height = env.scene[robot_str].data.body_link_pos_w[..., 8, 2]

    return torch.where((ee_height < 0.015) | (ee_height < 0.015), -1, 0)


def punish_bad_elbow_shoulder_posture(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    elbow_position = robot.data.joint_pos[:, 2]
    shoulder_position = robot.data.joint_pos[:, 1]
    # Punish for bad elbow position
    elbow_punishment = torch.where(elbow_position < 1.8, 1.8 - elbow_position, torch.tensor(0.0))
    elbow_punishment += torch.where(elbow_position > 2.5, elbow_position - 2.5, torch.tensor(0.0))
    # Punish for bad shoulder position (corrected to use shoulder_position)
    shoulder_punishment = torch.where(shoulder_position < 0.8, 0.8 - shoulder_position, torch.tensor(0.0))
    shoulder_punishment += torch.where(shoulder_position > 1.6, shoulder_position - 1.6, torch.tensor(0.0))
    # Combine punishments
    total_punishment = elbow_punishment + shoulder_punishment
    # Cap the total punishment at a maximum of 1
    total_punishment = torch.clamp(total_punishment, max=1)

    return -total_punishment


def from_eeframes_get_tip_mid_pos_w(fixed_chop_tip_frame, free_chop_tip_frame):
    ee_tip_fixed_w = fixed_chop_tip_frame.data.target_pos_w[..., 0, :]
    ee_tip_free_w = free_chop_tip_frame.data.target_pos_w[..., 0, :]
    mid_pos_w = (ee_tip_fixed_w + ee_tip_free_w) / 2.0
    return mid_pos_w


def get_objectFrame_eeFrames_distance(object_frame, fixed_chop_tip_frame, free_chop_tip_frame):
    # Target object_frame position: (num_envs, 3)
    object_frame_pos_w = object_frame.data.target_pos_w[..., 0, :]
    # End-effector aiming position: (num_envs, 3)
    tip_mid_pos_w = from_eeframes_get_tip_mid_pos_w(fixed_chop_tip_frame, free_chop_tip_frame)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(object_frame_pos_w - tip_mid_pos_w, dim=1)
    return object_ee_distance


def get_object_eeFrame_distance(object, offset, fixed_chop_tip_frame, free_chop_tip_frame):
    # End-effector aiming position: (num_envs, 3)
    mid_pos_w = from_eeframes_get_tip_mid_pos_w(fixed_chop_tip_frame, free_chop_tip_frame)
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm((object.data.root_pos_w + offset) - mid_pos_w, dim=1)
    return object_ee_distance


def link_orientation_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b = command[:, 3:7]
    des_quat_w = quat_mul(asset.data.root_state_w[:, 3:7], des_quat_b)
    curr_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0], 3:7]  # type: ignore
    return 1 - torch.tanh(quat_error_magnitude(curr_quat_w, des_quat_w) / std)
