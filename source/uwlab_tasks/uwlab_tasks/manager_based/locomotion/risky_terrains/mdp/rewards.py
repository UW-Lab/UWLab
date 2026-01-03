# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor


def joint_vel_limit_pen(
    env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"), limits_factor: float = 0.9
):
    robot: Articulation = env.scene[robot_cfg.name]
    joint_vel = robot.data.joint_vel[:, robot_cfg.joint_ids]
    joint_vel_limit = robot.data.soft_joint_vel_limits[:, robot_cfg.joint_ids]
    return torch.sum((joint_vel.abs() - limits_factor * joint_vel_limit).clamp_min_(0), dim=-1)


def base_accel_pen(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="base"),
    ratio: float = 0.02,
):
    robot: Articulation = env.scene[robot_cfg.name]
    base_angle_accel = robot.data.body_ang_acc_w[:, robot_cfg.body_ids].norm(2, dim=-1).pow(2)
    base_lin_accel = robot.data.body_lin_acc_w[:, robot_cfg.body_ids].norm(2, dim=-1).pow(2)
    return (base_lin_accel + ratio * base_angle_accel).squeeze(-1)


def feet_accel_l1_pen(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*FOOT"),
):
    robot: Articulation = env.scene[robot_cfg.name]
    feet_acc = robot.data.body_lin_acc_w[:, robot_cfg.body_ids].norm(2, dim=-1)
    return torch.sum(feet_acc, dim=-1)


def contact_forces_pen(
    env: ManagerBasedRLEnv, threshold: float = 700, sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_sensor")
):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    force = torch.norm(net_contact_forces[:, 0, sensor_cfg.body_ids], dim=-1)
    return torch.clamp(force - threshold, 0, threshold).pow(2).sum(-1)


def position_tracking(env: ManagerBasedRLEnv, tr: float = 2.0):
    desired_xy = env.command_manager.get_command("target_cmd")[:, :2]
    start_step = env.extensions["max_episode_length"] * (1 - tr / env.extensions["max_episode_length_s"])
    r_pos_tracking = 1 / tr * (1 / (1 + desired_xy.norm(2, -1).pow(2))) * (env.episode_length_buf > start_step).float()
    return r_pos_tracking


def heading_tracking(env: ManagerBasedRLEnv, d: float = 2.0, tr: float = 4.0):
    desired_heading = env.command_manager.get_command("target_cmd")[:, 3]
    start_step = env.extensions["max_episode_length"] * (1 - tr / env.extensions["max_episode_length_s"])
    current_dist = env.command_manager.get_command("target_cmd")[:, :2].norm(2, -1)
    r_heading_tracking = (
        1
        / tr
        * (1 / (1 + desired_heading.pow(2)))
        * (current_dist < d).float()
        * (env.episode_length_buf > start_step).float()
    )
    return r_heading_tracking


def dont_wait(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    d: float = 1.0,
    velocity_threshold: float = 0.2,
):
    robot: Articulation = env.scene[robot_cfg.name]
    dist_to_goal = env.command_manager.get_command("target_cmd")[:, :2].norm(2, -1)
    return (dist_to_goal > d).float() * (robot.data.root_lin_vel_w.norm(2, -1) < velocity_threshold).float()


def move_in_dir(
    env: ManagerBasedRLEnv,
    max_iter: int = 150,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot: Articulation = env.scene[robot_cfg.name]
    lin_vel_b = robot.data.root_lin_vel_b[:, :2]
    target_dir = env.command_manager.get_command("target_cmd")[:, :2]
    current_iter = int(env.common_step_counter / 48)
    return torch.cosine_similarity(lin_vel_b, target_dir, dim=-1) * float(current_iter < max_iter)


def foot_on_ground(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, d: float = 0.25, tr: float = 1.0):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    foot_on_ground_rew = 1 - torch.tanh(contact_sensor.data.current_air_time[:, sensor_cfg.body_ids].sum(-1) / 0.5)

    distance_succ_mask = (env.command_manager.get_command("target_cmd")[:, :2].norm(2, -1)) < d
    rew_window_scaler = (
        1
        / tr
        * (
            env.episode_length_buf
            > ((1 - tr / env.extensions["max_episode_length_s"]) * env.extensions["max_episode_length"])
        ).float()
    )

    return torch.where(distance_succ_mask, foot_on_ground_rew * rew_window_scaler, 0.0)


def stand_still(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    d: float = 0.25,
    phi: float = 0.5,
    tr: float = 1.0,
):
    robot: Articulation = env.scene[robot_cfg.name]
    movement_penalty = 2.5 * robot.data.root_lin_vel_w.norm(2, -1) + 1.0 * robot.data.root_ang_vel_w.norm(2, -1)

    heading_succ_mask = (env.command_manager.get_command("target_cmd")[:, 3].abs()) < phi
    distance_succ_mask = (env.command_manager.get_command("target_cmd")[:, :2].norm(2, -1)) < d
    rew_window_scaler = (
        1
        / tr
        * (
            env.episode_length_buf
            > ((1 - tr / env.extensions["max_episode_length_s"]) * env.extensions["max_episode_length"])
        ).float()
    )

    return torch.where(heading_succ_mask & distance_succ_mask, movement_penalty * rew_window_scaler, 0.0)


def illegal_contact_penalty(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg):
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    # check if any contact force exceeds the threshold
    return torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    ).float()


import os
import torch.nn.functional as F
from torch import nn


class RNDCuriosityNet(nn.Module):
    def __init__(self, in_dim, out_dim, n_hid):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_hid = n_hid

        self.fc1 = nn.Linear(in_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc3 = nn.Linear(n_hid, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


class CuriosityReward(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self.obs_dim: int = env.observation_manager.group_obs_dim["policy"][0]
        self.action_dim = env.action_manager.action_term_dim[0]
        self.M1 = RNDCuriosityNet(self.obs_dim + self.action_dim, 1, 128).to(env.device)  # prediction network
        self.M2 = RNDCuriosityNet(self.obs_dim + self.action_dim, 1, 256).to(env.device)  # frozen target network
        self.M1.apply(init_weights)
        self.M2.apply(init_weights)  # init weights of target network
        self.optimizer = torch.optim.Adam(self.M1.parameters(), lr=cfg.params.get("lr"))
        self.loss = nn.L1Loss(reduce=False)

    def __call__(self, env: ManagerBasedRLEnv, optimization_weight: float, lr: float) -> torch.Tensor:
        if "log_dir" in env.extensions:
            if env.common_step_counter == 1:
                if not os.path.exists(os.path.join(env.extensions["log_dir"], "RND")):
                    os.mkdir(os.path.join(env.extensions["log_dir"], "RND"))
                torch.save(self.M2.state_dict(), os.path.join(env.extensions["log_dir"], "RND", "M2.pth"))

            if (env.common_step_counter // 48) % 500 == 0:
                torch.save(self.M1.state_dict(), os.path.join(env.extensions["log_dir"], "RND", "M1.pth"))

        with torch.inference_mode(False):
            obs: torch.Tensor = (
                env.obs_buf["policy"] if hasattr(env, "obs_buf") else env.observation_manager.compute()["policy"]
            )
            action = env.action_manager.action
            predict_value = self.M1(torch.cat([obs.detach(), action.detach()], dim=-1))
            with torch.no_grad():
                target_value = self.M2(torch.cat([obs, action], dim=-1))
            reward = self.loss(predict_value, target_value)
            loss = (reward * optimization_weight).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return reward.squeeze(-1).detach().clamp_(-10, 10)


# balance beams


def aggressive_motion(
    env: ManagerBasedRLEnv,
    threshold: float = 1.0,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    robot: Articulation = env.scene[robot_cfg.name]
    horizontal_velocity = robot.data.root_lin_vel_w[:, :2].norm(2, -1)
    return (horizontal_velocity - threshold).pow(2) * (horizontal_velocity > threshold).float()


def stand_pos(
    env: ManagerBasedRLEnv,
    base_height: float = 0.6,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*FOOT"),
    tr: float = 1,
    phi: float = 0.5,
    d: float = 0.25,
):
    robot: Articulation = env.scene[robot_cfg.name]
    return (
        ((robot.data.root_pos_w[:, -1] - base_height).abs() + robot.data.projected_gravity_b[:, :2].pow(2).sum(-1))
        * ((env.command_manager.get_command("target_cmd")[:, :2].norm(2, -1)) < d).float()
        * (
            1
            / tr
            * (
                env.episode_length_buf
                > ((1 - tr / env.extensions["max_episode_length_s"]) * env.extensions["max_episode_length"])
            ).float()
        )
        * ((env.command_manager.get_command("target_cmd")[:, 3].abs()) < phi).float()
    )


def torque_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    actuator_name: str = "legs",
    ratio: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    computed_torque = asset.data.computed_torque[:, asset_cfg.joint_ids].abs()  # shape: [batch, joint]
    limits = ratio * asset.actuators.get(actuator_name).effort_limit
    out_of_limits = torch.clamp(computed_torque - limits, min=0)
    return torch.sum(out_of_limits, dim=1)


# ============ Spot tailored rewards ============


def torque_limits_knee(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), ratio: float = 1.0
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    computed_torque = asset.data.computed_torque[:, asset_cfg.joint_ids]  # shape: [batch, joint]
    applied_torque = asset.data.applied_torque[:, asset_cfg.joint_ids]
    out_of_limits = ratio * (computed_torque - applied_torque).abs()
    return torch.sum(out_of_limits, dim=1)


def reward_forward_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    forward_vector,
    max_iter: int = 150,
    init_amplification: float = 1.0,
    distance_threshold: float = 0.4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    root_lin_vel_b = asset.data.root_lin_vel_b
    forward_velocity = root_lin_vel_b * torch.tensor(forward_vector, device=env.device, dtype=root_lin_vel_b.dtype)
    forward_reward = torch.sum(forward_velocity, dim=1)
    current_iter = int(env.common_step_counter / 48)
    coeff = init_amplification if current_iter < max_iter else 1.0
    distance = torch.norm(env.command_manager.get_command("target_cmd")[:, :2], dim=1)
    return torch.where(distance > distance_threshold, torch.tanh(forward_reward / std) * coeff, 0.0)


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    # cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1).unsqueeze(dim=1).expand(-1, 4)
    body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1).unsqueeze(dim=1).expand(-1, 4)
    distance = torch.norm(env.command_manager.get_command("target_cmd")[:, :2], dim=1)
    reward = torch.where(
        (distance > 0.4) & (body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        max_iterations: int = 500,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        current_iter = int(env.common_step_counter / 48)
        if current_iter > max_iterations:
            return torch.zeros(self.num_envs, device=self.device)
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        # cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        distance = torch.norm(env.command_manager.get_command("target_cmd")[:, :2], dim=1)
        approach_gait_rew = torch.where(
            ((distance > 0.4) & (body_vel > self.velocity_threshold)), sync_reward * async_reward, 0.0
        )
        stance_rew = torch.where((distance < 0.4), 1 - torch.tanh(body_vel / 0.1), 0.0)
        return approach_gait_rew + stance_rew

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    distance = torch.norm(env.command_manager.get_command("target_cmd")[:, :2], dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(
        torch.logical_or(distance > 0.4, body_vel > velocity_threshold), reward, stand_still_scale * reward
    )
