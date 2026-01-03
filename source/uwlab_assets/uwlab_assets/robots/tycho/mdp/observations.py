# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import subtract_frame_transforms


def end_effector_pose_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_name="robot",
    fixed_chop_frame_name="frame_fixed_chop_tip",
    free_chop_frame_name="frame_free_chop_tip",
):
    robot: RigidObject = env.scene[robot_name]
    fixed_chop_frame: FrameTransformer = env.scene[fixed_chop_frame_name]
    free_chop_frame: FrameTransformer = env.scene[free_chop_frame_name]
    fixed_chop_frame_pos_b, fixed_chop_frame_quat_b = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        fixed_chop_frame.data.target_pos_w[..., 0, :],
        fixed_chop_frame.data.target_quat_w[..., 0, :],
    )

    free_chop_frame_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3],
        robot.data.root_state_w[:, 3:7],
        free_chop_frame.data.target_pos_w[..., 0, :],
    )

    return torch.cat((fixed_chop_frame_pos_b, free_chop_frame_pos_b, fixed_chop_frame_quat_b), dim=1)
