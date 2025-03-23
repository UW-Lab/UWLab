# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from uwlab.envs import DataManagerBasedRLEnv


def viewport_follow_robot(
    env: DataManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    target: RigidObject = env.scene[asset_cfg.name]
    target_pos = target.data.root_pos_w.clone()
    new_camera_pos = target_pos + torch.tensor([-1.0, 3.0, 2 * target_pos[0, 2]], device=target_pos.device)
    direction = target_pos - new_camera_pos
    direction[:, 2] -= 0.2 * target_pos[0, 2]
    if env.viewport_camera_controller is not None:
        env.viewport_camera_controller.update_view_location(
            eye=new_camera_pos[0].cpu().tolist(), lookat=target_pos[0].cpu().tolist()
        )


def update_joint_target_positions_to_current(env: DataManagerBasedRLEnv, env_ids: torch.Tensor | None, asset_name: str):
    asset: Articulation = env.scene[asset_name]
    joint_pos_target = asset.data.joint_pos
    asset.set_joint_position_target(joint_pos_target)


def reset_robot_to_default(
    env: DataManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Reset the scene to the default state specified in the scene configuration."""
    robot: Articulation = env.scene[robot_cfg.name]
    default_root_state = robot.data.default_root_state[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    # set into the physics simulation
    robot.write_root_state_to_sim(default_root_state, env_ids=env_ids)
    # obtain default joint positions
    default_joint_pos = robot.data.default_joint_pos[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel[env_ids].clone()
    # set into the physics simulation
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel, env_ids=env_ids)


def launch_view_port(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    view_portname: str,
    camera_path: str,
    viewport_size: tuple[int, int] = (640, 360),
    position: tuple[int, int] = (0, 0),
):
    if env.sim.has_gui():
        from isaacsim.core.utils.viewports import create_viewport_for_camera, get_viewport_names

        if view_portname not in get_viewport_names():
            create_viewport_for_camera(
                viewport_name=view_portname,
                camera_prim_path=camera_path,
                width=viewport_size[0],
                height=viewport_size[1],
                position_x=position[0],
                position_y=position[1],
            )
