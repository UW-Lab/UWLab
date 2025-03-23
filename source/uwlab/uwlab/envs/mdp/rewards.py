# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_frame1_frame2_distance(frame1, frame2):
    frames_distance = torch.norm(frame1.data.target_pos_w[..., 0, :] - frame2.data.target_pos_w[..., 0, :], dim=1)
    return frames_distance


def get_body1_body2_distance(body1, body2, body1_offset, body2_offset):
    bodys_distance = torch.norm((body1.data.root_pos_w + body1_offset) - (body2.data.root_pos_w + body2_offset), dim=1)
    return bodys_distance


def get_frame1_body2_distance(frame1, body2, body2_offset):
    distance = torch.norm(frame1.data.target_pos_w[..., 0, :] - (body2.data.root_pos_w + body2_offset), dim=1)
    return distance


def reward_body_height_above(env: ManagerBasedRLEnv, minimum_height: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when the asset's height is below the minimum height.

    Note:
        This is currently only supported for flat terrains, i.e. the minimum height is in the world frame.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.where(asset.data.root_pos_w[:, 2] > minimum_height, 1, 0)


def reward_frame1_frame2_distance(
    env: ManagerBasedRLEnv,
    frame1_cfg: SceneEntityCfg,
    frame2_cfg: SceneEntityCfg,
) -> torch.Tensor:
    object_frame1: FrameTransformer = env.scene[frame1_cfg.name]
    object_frame2: FrameTransformer = env.scene[frame2_cfg.name]
    frames_distance = get_frame1_frame2_distance(object_frame1, object_frame2)
    return 1 - torch.tanh(frames_distance / 0.1)


def reward_body1_body2_distance(
    env: ManagerBasedRLEnv,
    body1_cfg: SceneEntityCfg,
    body2_cfg: SceneEntityCfg,
    std: float,
    body1_offset: list[float] = [0.0, 0.0, 0.0],
    body2_offset: list[float] = [0.0, 0.0, 0.0],
) -> torch.Tensor:
    body1: RigidObject = env.scene[body1_cfg.name]
    body2: RigidObject = env.scene[body2_cfg.name]
    body1_offset_tensor = torch.tensor(body1_offset, device=env.device)
    body2_offset_tensor = torch.tensor(body2_offset, device=env.device)
    bodys_distance = get_body1_body2_distance(body1, body2, body1_offset_tensor, body2_offset_tensor)

    return 1 - torch.tanh(bodys_distance / std)


def reward_body1_frame2_distance(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    frame_cfg: SceneEntityCfg,
    body_offset: list[float] = [0.0, 0.0, 0.0],
) -> torch.Tensor:
    body: RigidObject = env.scene[body_cfg.name]
    object_frame: RigidObject = env.scene[frame_cfg.name]
    body_offset_tensor = torch.tensor(body_offset, device=env.device)
    bodys_distance = get_frame1_body2_distance(object_frame, body, body_offset_tensor)
    return 1 - torch.tanh(bodys_distance / 0.1)


def reward_body1_body2_within_distance(
    env: ManagerBasedRLEnv,
    body1_cfg: SceneEntityCfg,
    body2_cfg: SceneEntityCfg,
    min_distance: float,
    body1_offset: list[float] = [0.0, 0.0, 0.0],
    body2_offset: list[float] = [0.0, 0.0, 0.0],
) -> torch.Tensor:
    body1: RigidObject = env.scene[body1_cfg.name]
    body2: RigidObject = env.scene[body2_cfg.name]
    body1_offset_tensor = torch.tensor(body1_offset, device=env.device)
    body2_offset_tensor = torch.tensor(body2_offset, device=env.device)
    bodys_distance = get_body1_body2_distance(body1, body2, body1_offset_tensor, body2_offset_tensor)
    reward = torch.where(bodys_distance < min_distance, 1.0, 0.0)
    return reward


def reward_being_alive(
    env: ManagerBasedRLEnv,
) -> torch.Tensor:
    return torch.tanh((env.episode_length_buf / env.max_episode_length) / 0.1)


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2-kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.where(env.episode_length_buf > 20, torch.square(asset.data.root_lin_vel_b[:, 2]), 0)


def l2_norm_joint_position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    cur_joint_position = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.norm(command - cur_joint_position, dim=1)
    return error


def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], des_pos_b)
    curr_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids[0], :3]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error_tanh(
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
    curr_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return 1 - torch.tanh(quat_error_magnitude(curr_quat_w, des_quat_w) / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
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
    curr_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)
