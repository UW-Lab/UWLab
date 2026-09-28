# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def reset_robot_to_default(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Reset the scene to the default state specified in the scene configuration."""
    robot: Articulation = env.scene[robot_cfg.name]
    default_root_state = robot.data.default_root_state.torch[env_ids].clone()
    default_root_state[:, 0:3] += env.scene.env_origins[env_ids]
    # set into the physics simulation
    robot.write_root_state_to_sim(default_root_state, env_ids=env_ids)
    # obtain default joint positions
    default_joint_pos = robot.data.default_joint_pos.torch[env_ids].clone()
    default_joint_vel = robot.data.default_joint_vel.torch[env_ids].clone()
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
        import omni.kit.commands
        from omni.kit.viewport.utility import create_viewport_window
        from omni.kit.viewport.window import get_viewport_window_instances

        if view_portname not in [window.title for window in get_viewport_window_instances()]:
            viewport_window = create_viewport_window(
                name=view_portname,
                width=viewport_size[0],
                height=viewport_size[1],
                position_x=position[0],
                position_y=position[1],
            )
            omni.kit.commands.execute(
                "SetViewportCamera", camera_path=camera_path, viewport_api=viewport_window.viewport_api
            )
