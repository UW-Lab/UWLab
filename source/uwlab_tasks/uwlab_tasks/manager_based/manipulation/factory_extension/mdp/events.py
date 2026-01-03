# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

from uwlab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from ..assembly_keypoints import KEYPOINTS_NISTBOARD

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from ..assembly_keypoints import Offset

# viz for debug, remove when done debugging
# from isaaclab.markers import FRAME_MARKER_CFG, VisualizationMarkers
# frame_marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
# frame_marker_cfg.markers["frame"].scale = (0.025, 0.025, 0.025)
# pose_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/debug_transform"))


def reset_fixed_assets(env: ManagerBasedRLEnv, env_ids: torch.tensor, asset_list: list[str]):
    nistboard: RigidObject = env.scene["nistboard"]
    for asset_str in asset_list:
        asset: Articulation | RigidObject = env.scene[asset_str]
        asset_offset_on_nist_board: Offset = getattr(KEYPOINTS_NISTBOARD, asset_str)
        asset_on_board_pos, asset_on_board_quat = asset_offset_on_nist_board.apply(nistboard)
        root_pose = torch.cat((asset_on_board_pos, asset_on_board_quat), dim=1)[env_ids]
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros_like(asset.data.root_vel_w[env_ids]), env_ids=env_ids)


def reset_held_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    holding_body_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
    held_asset_graspable_offset: Offset,
    held_asset_inhand_range: dict[str, tuple[float, float]],
):
    robot: Articulation = env.scene[holding_body_cfg.name]
    held_asset: Articulation = env.scene[held_asset_cfg.name]

    end_effector_quat_w = robot.data.body_link_quat_w[env_ids, holding_body_cfg.body_ids].view(-1, 4)
    end_effector_pos_w = robot.data.body_link_pos_w[env_ids, holding_body_cfg.body_ids].view(-1, 3)
    held_graspable_pos_b = torch.tensor(held_asset_graspable_offset.pos, device=env.device).repeat(len(env_ids), 1)
    held_graspable_quat_b = torch.tensor(held_asset_graspable_offset.quat, device=env.device).repeat(len(env_ids), 1)

    flip_z_quat = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=env.device).repeat(len(env_ids), 1)
    translated_held_asset_pos, translated_held_asset_quat = _pose_a_when_frame_ba_aligns_pose_c(
        pos_c=end_effector_pos_w,
        quat_c=math_utils.quat_mul(end_effector_quat_w, flip_z_quat),
        pos_ba=held_graspable_pos_b,
        quat_ba=held_graspable_quat_b,
    )

    # Add randomization
    range_list = [held_asset_inhand_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=env.device)
    samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=env.device)
    new_pos_w = translated_held_asset_pos + samples[:, 0:3]
    quat_b = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
    new_quat_w = math_utils.quat_mul(translated_held_asset_quat, quat_b)

    held_asset.write_root_link_pose_to_sim(torch.cat([new_pos_w, new_quat_w], dim=1), env_ids=env_ids)  # type: ignore
    held_asset.write_root_com_velocity_to_sim(held_asset.data.default_root_state[env_ids, 7:], env_ids=env_ids)  # type: ignore


def grasp_held_asset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    held_asset_diameter: float,
) -> None:
    robot: Articulation = env.scene[robot_cfg.name]
    joint_pos = robot.data.joint_pos[:, robot_cfg.joint_ids][env_ids].clone()
    joint_pos[:, :] = held_asset_diameter / 2 * 1.25
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), robot_cfg.joint_ids, env_ids)  # type: ignore


class reset_end_effector_round_fixed_asset(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        fixed_asset_cfg: SceneEntityCfg = cfg.params.get("fixed_asset_cfg")  # type: ignore
        fixed_asset_offset: Offset = cfg.params.get("fixed_asset_offset")  # type: ignore
        pose_range_b: dict[str, tuple[float, float]] = cfg.params.get("pose_range_b")  # type: ignore
        robot_ik_cfg: SceneEntityCfg = cfg.params.get("robot_ik_cfg", SceneEntityCfg("robot"))

        range_list = [pose_range_b.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.ranges = torch.tensor(range_list, device=env.device)
        self.fixed_asset: Articulation | RigidObject = env.scene[fixed_asset_cfg.name]
        self.fixed_asset_offset: Offset = fixed_asset_offset
        self.robot: Articulation = env.scene[robot_ik_cfg.name]
        self.joint_ids: list[int] | slice = robot_ik_cfg.joint_ids
        self.robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=robot_ik_cfg.name,
            joint_names=robot_ik_cfg.joint_names,  # type: ignore
            body_name=robot_ik_cfg.body_names,  # type: ignore
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = None  # type: ignore

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        fixed_asset_cfg: SceneEntityCfg,
        fixed_asset_offset: Offset,
        pose_range_b: dict[str, tuple[float, float]],
        robot_ik_cfg: SceneEntityCfg,
    ) -> None:
        if self.solver is None:
            self.solver = self.robot_ik_solver_cfg.class_type(self.robot_ik_solver_cfg, env)
        fixed_tip_pos_w, fixed_tip_quat_w = self.fixed_asset_offset.apply(self.fixed_asset)
        samples = math_utils.sample_uniform(self.ranges[:, 0], self.ranges[:, 1], (len(env_ids), 6), device=env.device)
        pos_b, quat_b = self.solver._compute_frame_pose()
        # for those non_reset_id, we will let ik solve for its current position
        pos_w = fixed_tip_pos_w[env_ids] + samples[:, 0:3]
        quat_w = math_utils.quat_from_euler_xyz(samples[:, 3], samples[:, 4], samples[:, 5])
        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            self.robot.data.root_link_pos_w[env_ids], self.robot.data.root_link_quat_w[env_ids], pos_w, quat_w
        )
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))
        n_joints: int = self.robot.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)
        # Error Rate 75% ^ 10 = 0.05 (final error)
        for i in range(10):
            self.solver.apply_actions()
            delta_joint_pos = 0.25 * (self.robot.data.joint_pos_target[env_ids] - self.robot.data.joint_pos[env_ids])
            self.robot.write_joint_state_to_sim(
                position=(delta_joint_pos + self.robot.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), n_joints), device=env.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,  # type: ignore
            )


def _pose_a_when_frame_ba_aligns_pose_c(
    pos_c: torch.Tensor, quat_c: torch.Tensor, pos_ba: torch.Tensor, quat_ba: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # TA←W ​= {TB←A}-1 ​∘ TC←W​   where  ​combine_transform(a,b): b∘a
    inv_pos_ba = -math_utils.quat_apply(math_utils.quat_inv(quat_ba), pos_ba)
    inv_quat_ba = math_utils.quat_inv(quat_ba)
    return math_utils.combine_frame_transforms(pos_c, quat_c, inv_pos_ba, inv_quat_ba)
