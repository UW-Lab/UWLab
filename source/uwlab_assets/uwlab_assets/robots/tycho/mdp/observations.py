# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import convert_dict_to_backend
from isaaclab.utils.math import subtract_frame_transforms
from uwlab.envs import DataManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    offset: list[float] = [0.0, 0.0, 0.0],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3].clone()
    object_pos_w[:] += torch.tensor(offset, device=env.device)
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_pos_w
    )
    return object_pos_b


def position_in_robot_root_frame(env: DataManagerBasedRLEnv, position_b: str) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    return env.data_manager.get_active_term("data", position_b)


def object_frame_position_in_robot_root_frame(
    env: DataManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_frame_cfg: SceneEntityCfg = SceneEntityCfg("object_frame"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object_frame: FrameTransformer = env.scene[object_frame_cfg.name]
    object_frame_pos_w = object_frame.data.target_pos_w[..., 0, :]
    object_frame_pos_b, _ = subtract_frame_transforms(
        robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], object_frame_pos_w
    )
    return object_frame_pos_b


def end_effector_speed(env: DataManagerBasedRLEnv, end_effector_speed_str: str) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    return env.data_manager.get_active_term("data", end_effector_speed_str)


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


def capture_image(env: ManagerBasedRLEnv, camera_key: str):
    futures = []
    for i in range(env.num_envs):
        # Launch parallel tasks
        future = torch.jit.fork(_process_camera_data, env, camera_key, i)
        futures.append(future)

    # Wait for all tasks to complete and collect results
    results = [torch.jit.wait(f) for f in futures]

    # Concatenate the results along the specified dimension
    obs_cam_data = torch.stack(results, dim=0)
    return obs_cam_data


def _process_camera_data(env, camera_key, idx):
    """
    Process camera data for a single environment.
    This function is intended to be executed in parallel for each environment.
    """
    camera = env.scene[camera_key]
    camera.update(dt=env.sim.get_physics_dt())
    cam_data = convert_dict_to_backend(camera.data.output[idx], backend="torch")
    cam_data_rgb = cam_data["rgb"][:, :, :3].contiguous()
    return cam_data_rgb.view(-1)
