# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers.recorder_manager import RecorderTerm


class StableStateRecorder(RecorderTerm):
    def record_pre_reset(self, env_ids):
        def extract_env_ids_values(value):
            nonlocal env_ids
            if isinstance(value, dict):
                return {k: extract_env_ids_values(v) for k, v in value.items()}
            return value[env_ids]

        return "initial_state", extract_env_ids_values(self._env.scene.get_state(is_relative=True))


class GraspRelativePoseRecorder(RecorderTerm):
    """Recorder term that records relative position, orientation, and gripper joint states for grasp evaluation."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        # Configuration for which robot and object to track
        self.robot_name = cfg.robot_name
        self.object_name = cfg.object_name
        self.gripper_body_name = cfg.gripper_body_name

    def record_pre_reset(self, env_ids):
        """Record relative pose between object and gripper, plus gripper joint states before reset."""
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.tensor(env_ids, device=self._env.device)

        # Get robot articulation and object rigid body
        robot = self._env.scene[self.robot_name]
        obj = self._env.scene[self.object_name]

        # Get object pose (root pose contains position and orientation)
        obj_root_state = obj.data.root_state_w[env_ids]  # Shape: (num_envs, 13) - pos(3) + quat(4) + vel(6)
        obj_pos = obj_root_state[:, :3]  # Position
        obj_quat = obj_root_state[:, 3:7]  # Quaternion (w, x, y, z)

        # Get gripper body pose from the robot articulation
        # Find the gripper body index
        gripper_body_idx = None
        for idx, body_name in enumerate(robot.body_names):
            if self.gripper_body_name in body_name:
                gripper_body_idx = idx
                break

        # Get specific body pose
        gripper_pos = robot.data.body_state_w[env_ids, gripper_body_idx, :3]
        gripper_quat = robot.data.body_state_w[env_ids, gripper_body_idx, 3:7]

        # Calculate relative transform: T_gripper_in_object = T_object^{-1} * T_gripper
        relative_pos, relative_quat = math_utils.subtract_frame_transforms(obj_pos, obj_quat, gripper_pos, gripper_quat)

        # Get gripper joint states as dict mapping joint names to positions
        gripper_joint_pos = robot.data.joint_pos[env_ids].clone()
        gripper_joint_dict = {joint_name: gripper_joint_pos[:, i] for i, joint_name in enumerate(robot.joint_names)}

        # Prepare data to record
        grasp_data = {
            "relative_position": relative_pos,
            "relative_orientation": relative_quat,
            "gripper_joint_positions": gripper_joint_dict,
        }

        return "grasp_relative_pose", grasp_data
