# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Reference: [Advanced Skills by Learning Locomotion and Local Navigation End-to-End, Nikita Rudin(s),
#             https://arxiv.org/pdf/2209.12827]

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def task_reward(env: ManagerBasedRLEnv, reward_window: float = 1.0):  # Represents Tr, the length of the reward window
    #
    # See section II.B (page 3) Exploration Reward for details.
    # Calculate the time step at which the reward window starts
    reward_start_step = env.max_episode_length * (1 - reward_window / env.max_episode_length_s)

    # Calculate the distance to the goal (‖xb − x∗b‖^2), squared L2 norm of the difference
    distance_to_goal = env.command_manager.get_command("goal_point")[:, :2].norm(2, -1).pow(2)

    # Calculate task reward as per the equation:
    # If within the reward window, r_task is non-zero
    task_reward = (1 / (1 + distance_to_goal)) * (env.episode_length_buf > reward_start_step).float()
    residue_task_reward = (1 / reward_window) * task_reward

    # TODO: Try no to change exploration weight here.
    # "The following line removes the exploration reward (r_bias) once the task reward (r_task)
    #  reaches 50% of its maximum value, as described in the paper." [II.B (page 3)]
    if task_reward.mean() > 0.5 and (env.reward_manager.get_term_cfg("exploration").weight > 0.0):
        env.reward_manager.get_term_cfg("exploration").weight = 0.0

    return residue_task_reward


def heading_tracking(env: ManagerBasedRLEnv, distance_threshold: float = 2.0, reward_window: float = 2.0):
    desired_heading = env.command_manager.get_command("goal_point")[:, 3]
    reward_start_step = env.max_episode_length * (1 - reward_window / env.max_episode_length_s)
    current_dist = env.command_manager.get_command("goal_point")[:, :2].norm(2, -1)
    r_heading_tracking = (
        1
        / reward_window
        * (1 / (1 + desired_heading.pow(2)))
        * (current_dist < distance_threshold).float()
        * (env.episode_length_buf > reward_start_step).float()
    )
    return r_heading_tracking


def exploration_reward(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    # Retrieve the robot and target data
    robot: Articulation = env.scene[robot_cfg.name]
    base_velocity = robot.data.root_lin_vel_b  # Robot's current base velocity vector
    target_position = env.command_manager.get_command("goal_point")[
        :, :2
    ]  # Target position vector relative to robot base

    # Compute the dot product of the robot's base velocity and target position vectors
    velocity_alignment = (base_velocity[:, :2] * target_position).sum(-1)

    # Calculate the norms (magnitudes) of the velocity and target position vectors
    velocity_magnitude = torch.norm(base_velocity, p=2, dim=-1)
    target_magnitude = torch.norm(target_position, p=2, dim=-1)

    # Calculate the exploration reward by normalizing the dot product (cosine similarity)
    # Small epsilon added in denominator to prevent division by zero
    exploration_reward = velocity_alignment / (velocity_magnitude * target_magnitude + 1e-6)
    return exploration_reward


def stall_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_vel_threshold: float = 0.1,
    distance_threshold: float = 0.5,
):
    robot: Articulation = env.scene[robot_cfg.name]
    base_vel = robot.data.root_lin_vel_b.norm(2, dim=-1)
    distance_to_goal = env.command_manager.get_command("goal_point")[:, :2].norm(2, dim=-1)
    return (base_vel < base_vel_threshold) & (distance_to_goal > distance_threshold)


def illegal_contact_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]  # type: ignore
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,  # type: ignore
    ).float()


def feet_lin_acc_l2(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    feet_acc = torch.sum(torch.square(robot.data.body_lin_acc_w[..., robot_cfg.body_ids, :]), dim=(1, 2))
    return feet_acc


def feet_rot_acc_l2(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    robot: Articulation = env.scene[robot_cfg.name]
    feet_acc = torch.sum(torch.square(robot.data.body_ang_acc_w[..., robot_cfg.body_ids, :]), dim=(1, 2))
    return feet_acc


def stand_penalty(
    env: ManagerBasedRLEnv,
    height_threshold: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    base_height = robot.data.root_link_pos_w[:, 2]  # z-coordinate of the base
    penalty = (base_height < height_threshold).float() * -1.0
    return penalty
