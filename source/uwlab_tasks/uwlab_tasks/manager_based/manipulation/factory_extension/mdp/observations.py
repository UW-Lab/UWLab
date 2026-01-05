# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from ..assembly_keypoints import Offset


def target_asset_pose_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_asset_offset: Offset | None = None,
    root_asset_offset: Offset | None = None,
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    taget_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids
    root_body_idx = 0 if isinstance(root_asset_cfg.body_ids, slice) else root_asset_cfg.body_ids

    target_pos = target_asset.data.body_link_pos_w[:, taget_body_idx].view(-1, 3)
    target_quat = target_asset.data.body_link_quat_w[:, taget_body_idx].view(-1, 4)
    root_pos = root_asset.data.body_link_pos_w[:, root_body_idx].view(-1, 3)
    root_quat = root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

    if root_asset_offset is not None:
        root_pos, root_quat = root_asset_offset.combine(root_pos, root_quat)
    if target_asset_offset is not None:
        target_pos, target_quat = target_asset_offset.combine(target_pos, target_quat)

    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)
    return torch.cat([target_pos_b, target_quat_b], dim=1)


def asset_link_velocity_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids

    asset_lin_vel_b, _ = math_utils.subtract_frame_transforms(
        root_asset.data.root_pos_w,
        root_asset.data.root_quat_w,
        target_asset.data.body_lin_vel_w[:, target_body_idx].view(-1, 3),
    )
    asset_ang_vel_b, _ = math_utils.subtract_frame_transforms(
        root_asset.data.root_pos_w,
        root_asset.data.root_quat_w,
        target_asset.data.body_ang_vel_w[:, target_body_idx].view(-1, 3),
    )

    return torch.cat([asset_lin_vel_b, asset_ang_vel_b], dim=1)
