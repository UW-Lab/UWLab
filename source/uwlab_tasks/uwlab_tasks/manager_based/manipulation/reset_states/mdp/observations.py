# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

from uwlab_tasks.manager_based.manipulation.reset_states.assembly_keypoints import Offset
from uwlab_tasks.manager_based.manipulation.reset_states.mdp import utils


def target_asset_pose_in_root_asset_frame(
    env: ManagerBasedEnv,
    target_asset_cfg: SceneEntityCfg,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_asset_offset=None,
    root_asset_offset=None,
    rotation_repr: str = "quat",
):
    target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
    root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]

    target_body_idx = 0 if isinstance(target_asset_cfg.body_ids, slice) else target_asset_cfg.body_ids
    root_body_idx = 0 if isinstance(root_asset_cfg.body_ids, slice) else root_asset_cfg.body_ids

    target_pos = target_asset.data.body_link_pos_w[:, target_body_idx].view(-1, 3)
    target_quat = target_asset.data.body_link_quat_w[:, target_body_idx].view(-1, 4)
    root_pos = root_asset.data.body_link_pos_w[:, root_body_idx].view(-1, 3)
    root_quat = root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

    if root_asset_offset is not None:
        root_pos, root_quat = root_asset_offset.combine(root_pos, root_quat)
    if target_asset_offset is not None:
        target_pos, target_quat = target_asset_offset.combine(target_pos, target_quat)

    target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)

    if rotation_repr == "axis_angle":
        axis_angle = math_utils.axis_angle_from_quat(target_quat_b)
        return torch.cat([target_pos_b, axis_angle], dim=1)
    elif rotation_repr == "quat":
        return torch.cat([target_pos_b, target_quat_b], dim=1)
    else:
        raise ValueError(f"Invalid rotation_repr: {rotation_repr}. Must be one of: 'quat', 'axis_angle'")


class target_asset_pose_in_root_asset_frame_with_metadata(ManagerTermBase):
    """Get target asset pose in root asset frame with offsets automatically read from metadata.

    This is similar to target_asset_pose_in_root_asset_frame but automatically reads the
    assembled offsets from the asset USD metadata instead of requiring manual specification.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        target_asset_cfg: SceneEntityCfg = cfg.params.get("target_asset_cfg")
        root_asset_cfg: SceneEntityCfg = cfg.params.get("root_asset_cfg", SceneEntityCfg("robot"))
        target_asset_offset_metadata_key: str = cfg.params.get("target_asset_offset_metadata_key")
        root_asset_offset_metadata_key: str = cfg.params.get("root_asset_offset_metadata_key")

        self.target_asset: RigidObject | Articulation = env.scene[target_asset_cfg.name]
        self.root_asset: RigidObject | Articulation = env.scene[root_asset_cfg.name]
        self.target_asset_cfg = target_asset_cfg
        self.root_asset_cfg = root_asset_cfg
        self.rotation_repr = cfg.params.get("rotation_repr", "quat")

        # Read root asset offset from metadata
        if root_asset_offset_metadata_key is not None:
            root_usd_path = self.root_asset.cfg.spawn.usd_path
            root_metadata = utils.read_metadata_from_usd_directory(root_usd_path)
            root_offset_data = root_metadata.get(root_asset_offset_metadata_key)
            self.root_asset_offset = Offset(pos=root_offset_data.get("pos"), quat=root_offset_data.get("quat"))
        else:
            self.root_asset_offset = None

        # Read target asset offset from metadata
        if target_asset_offset_metadata_key is not None:
            target_usd_path = self.target_asset.cfg.spawn.usd_path
            target_metadata = utils.read_metadata_from_usd_directory(target_usd_path)
            target_offset_data = target_metadata.get(target_asset_offset_metadata_key)
            self.target_asset_offset = Offset(pos=target_offset_data.get("pos"), quat=target_offset_data.get("quat"))
        else:
            self.target_asset_offset = None

    def __call__(
        self,
        env: ManagerBasedEnv,
        target_asset_cfg: SceneEntityCfg,
        root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        target_asset_offset_metadata_key: str | None = None,
        root_asset_offset_metadata_key: str | None = None,
        rotation_repr: str = "quat",
    ) -> torch.Tensor:
        target_body_idx = 0 if isinstance(self.target_asset_cfg.body_ids, slice) else self.target_asset_cfg.body_ids
        root_body_idx = 0 if isinstance(self.root_asset_cfg.body_ids, slice) else self.root_asset_cfg.body_ids

        target_pos = self.target_asset.data.body_link_pos_w[:, target_body_idx].view(-1, 3)
        target_quat = self.target_asset.data.body_link_quat_w[:, target_body_idx].view(-1, 4)
        root_pos = self.root_asset.data.body_link_pos_w[:, root_body_idx].view(-1, 3)
        root_quat = self.root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

        if self.root_asset_offset is not None:
            root_pos, root_quat = self.root_asset_offset.combine(root_pos, root_quat)
        if self.target_asset_offset is not None:
            target_pos, target_quat = self.target_asset_offset.combine(target_pos, target_quat)

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, target_pos, target_quat)

        if rotation_repr == "axis_angle":
            axis_angle = math_utils.axis_angle_from_quat(target_quat_b)
            return torch.cat([target_pos_b, axis_angle], dim=1)
        elif rotation_repr == "quat":
            return torch.cat([target_pos_b, target_quat_b], dim=1)
        else:
            raise ValueError(f"Invalid rotation_repr: {rotation_repr}. Must be one of: 'quat', 'axis_angle'")


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


def get_material_properties(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_material_properties().view(env.num_envs, -1)


def get_mass(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.root_physx_view.get_masses().view(env.num_envs, -1)


def get_joint_friction(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_friction_coeff.view(env.num_envs, -1)


def get_joint_armature(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_armature.view(env.num_envs, -1)


def get_joint_stiffness(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_stiffness.view(env.num_envs, -1)


def get_joint_damping(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
):
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_damping.view(env.num_envs, -1)


def time_left(env) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        life_left = 1 - (env.episode_length_buf.float() / env.max_episode_length)
    else:
        life_left = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    return life_left.view(-1, 1)
