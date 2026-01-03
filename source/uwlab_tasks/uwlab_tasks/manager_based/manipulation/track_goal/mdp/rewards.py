# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def stay_still(
    env: ManagerBasedRLEnv,
    ee_command_name: str,
    hand_command_name: str,
    hand_asset_cfg: SceneEntityCfg,
    ee_asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[hand_asset_cfg.name]
    ee_command = env.command_manager.get_command(ee_command_name)
    hand_command = env.command_manager.get_command(hand_command_name)
    cur_hand_joint_pos = asset.data.joint_pos[:, hand_asset_cfg.joint_ids]
    hand_error = torch.norm(hand_command - cur_hand_joint_pos, dim=1)
    des_pos_w, _ = math_utils.combine_frame_transforms(
        asset.data.root_state_w[:, :3], asset.data.root_state_w[:, 3:7], ee_command[:, :3]
    )
    curr_pos_w = asset.data.body_link_pos_w[:, ee_asset_cfg.body_ids, :3].view(-1, 3)
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    goal_reached_mask = (distance < 0.1) & (hand_error < 0.4)

    return torch.where(goal_reached_mask, asset.data.body_vel_w.abs().sum(2).sum(1), 0)


def delta_action_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    delta_action = env.action_manager.action - env.action_manager.prev_action
    return torch.sum(torch.square(delta_action), dim=1)
