# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for hand joint tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .command_cfg import HandJointCommandCfg


class HandJointCommand(CommandTerm):
    cfg: HandJointCommandCfg

    def __init__(self, cfg: HandJointCommandCfg, env: ManagerBasedRLEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        self.env = env
        super().__init__(cfg, env)
        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_cfg.name]
        self.joint_indices = self.robot.find_joints(cfg.asset_cfg.joint_names)[0]  # type: ignore
        self.predefined_hand_joint_goals = torch.tensor(cfg.predefined_hand_joint_goals, device=self.device)
        self.current_joint_goal = torch.zeros((self.num_envs, len(self.joint_indices)), device=self.device)

        r = torch.randint(0, len(self.predefined_hand_joint_goals), (self.num_envs,))
        self.current_joint_goal[:] = self.predefined_hand_joint_goals[r]
        # -- metrics
        self.metrics["joint_position_error"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.current_joint_goal

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        current_joint_pos = self.robot.data.joint_pos[:, self.joint_indices]
        self.metrics["position_error"] = torch.norm(self.current_joint_goal - current_joint_pos, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new joint targets

        r = torch.randint(0, len(self.predefined_hand_joint_goals), (len(env_ids),))
        self.current_joint_goal[env_ids] = self.predefined_hand_joint_goals[r]

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdGeom

        if debug_vis:
            if not hasattr(self, "vis_articulation"):
                if self.cfg.articulation_vis_cfg.name in self.env.scene.keys():  # noqa: SIM118
                    self.vis_articulation: Articulation = self.env.scene[self.cfg.articulation_vis_cfg.name]
                    prims_paths = prim_utils.find_matching_prim_paths(self.vis_articulation.cfg.prim_path)
                    prims = [prim_utils.get_prim_at_path(prim) for prim in prims_paths]
                    for prim in prims:
                        UsdGeom.Imageable(prim).MakeVisible()
                # VisualizationMarkers
        else:
            if hasattr(self, "vis_articulation"):
                prims_paths = prim_utils.find_matching_prim_paths(self.vis_articulation.cfg.prim_path)
                prims = [prim_utils.get_prim_at_path(prim) for prim in prims_paths]
                for prim in prims:
                    UsdGeom.Imageable(prim).MakeInvisible()

    def _debug_vis_callback(self, event):
        # update the box marker
        writ_pose_command: CommandTerm = self.env.command_manager.get_term(self.cfg.wrist_pose_term)
        pose_command_b = writ_pose_command.command
        pos_command_w, quat_command_w = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            pose_command_b[:, :3],
            pose_command_b[:, 3:],
        )
        self.vis_articulation.write_root_pose_to_sim(torch.cat([pos_command_w, quat_command_w], dim=1))
        self.vis_articulation.write_joint_state_to_sim(
            position=self.current_joint_goal, velocity=torch.zeros(self.current_joint_goal.shape, device=self.device)
        )
