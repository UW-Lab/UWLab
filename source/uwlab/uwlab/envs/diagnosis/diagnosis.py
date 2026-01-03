# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_link_incoming_joint_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    This function provides force and torque exerted from child link to parent link in world frame
    This force/torque is in WorldFrame -> https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/Joints.html
    This force/torque is exerted by the joint connecting child link to the parent link ->
    https://docs.omniverse.nvidia.com/isaacsim/latest/features/sensors_simulation/isaac_sim_sensors_physics_articulation_force.html

    Return measured joint forces and torques. Shape is (num_articulation, num_joint + 1, 6). Row index 0 is the incoming
    joint of the base link. For the last dimension the first 3 values are for forces and the last 3 for torques
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    force_from_child_link_to_joints = asset.root_physx_view.get_link_incoming_joint_force().to(env.device)[env_ids]
    return force_from_child_link_to_joints


def get_dof_projected_joint_forces(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    get_measured_joint_efforts specifies the active component (the projection of the joint forces on the
    motion direction) of the joint force for all the joints and articulations.

    My guess: get_measured_joint_efforts() provides the actual active efforts acting along the joint axes during the simulation.
    It reflects the effective efforts being applied to the joints, considering factors such as: Actuator limitations,
    Dynamics of the system, External forces and constraints, Interactions with the environment

    Return dimension: (num_articulations, num_links)

    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    projected_joint_forces = asset.root_physx_view.get_dof_projected_joint_forces().to(env.device)[env_ids]
    return projected_joint_forces


def get_dof_applied_torque(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    get the torque that joint drive applies
    Return dimension: (num_articulations, num_links)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    applied_torque = asset.data.applied_torque[env_ids]
    return applied_torque


def get_dof_computed_torque(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    computed_torque = asset.data.computed_torque[env_ids]
    return computed_torque


def get_dof_target_position(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    target_joint_pos = asset.data.joint_pos_target[env_ids]
    return target_joint_pos


def get_dof_position(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    joint_pos = asset.data.joint_pos[env_ids]
    return joint_pos


def get_dof_target_velocity(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    target_joint_vel = asset.data.joint_vel_target[env_ids]
    return target_joint_vel


def get_dof_velocity(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    joint_vel = asset.data.joint_vel[env_ids]
    return joint_vel


def get_dof_acceleration(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    joint_acc = asset.data.joint_acc[env_ids]
    return joint_acc


def get_action_rate(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
) -> torch.Tensor:
    return torch.square(env.action_manager.action[env_ids] - env.action_manager.prev_action[env_ids])


def get_joint_torque_utilization(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    applied_torque = asset.data.applied_torque[env_ids]
    torque_max = asset.root_physx_view.get_dof_max_forces().to(env.device)[env_ids]
    torque_utilization = torch.abs(applied_torque) / torque_max
    return torque_utilization


def get_joint_velocity_utilization(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    joint_vel = asset.data.joint_vel[env_ids]
    max_vel = asset.root_physx_view.get_dof_max_velocities().to(env.device)[env_ids]
    velocity_utilization = torch.abs(joint_vel) / max_vel
    return velocity_utilization


def get_joint_power(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    joint_vel = asset.data.joint_vel[env_ids]
    applied_torque = asset.data.applied_torque[env_ids]
    power = torch.abs(applied_torque * joint_vel)
    return power


def get_joint_mechanical_work(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    joint_pos = asset.data.joint_pos[env_ids]
    applied_torque = asset.data.applied_torque[env_ids]
    if "prev_joint_pos" in env.extensions:
        delta_joint_pos = joint_pos - env.extensions["prev_joint_pos"]
        mechanical_work = torch.abs(delta_joint_pos * applied_torque)
        env.extensions["prev_joint_pos"] = joint_pos
        return mechanical_work
    else:
        env.extensions["prev_joint_pos"] = joint_pos
        return torch.zeros(joint_pos.shape, device=env.device)


def effective_torque(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Calculate the ratio of projected joint forces over applied joint forces for each joint.

    Args:
        env: The simulation environment.
        env_ids: Environment IDs (optional, not used here).
        asset_cfg: Asset configuration.

    Returns:
        force_ratio: Tensor of the ratio of projected to applied joint forces.
                     Shape: (num_envs, num_joints)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    # Get applied joint torques (from actuators)
    applied_torque = asset.data.applied_torque[env_ids]  # Shape: (num_envs, num_joints)

    # Get projected joint forces
    projected_joint_forces = asset.root_physx_view.get_dof_projected_joint_forces().to(env.device)[
        env_ids
    ]  # Shape: (num_envs, num_joints)

    # Handle zero applied torques to avoid division by zero
    epsilon = 1e-6  # Small constant
    applied_torque[torch.abs(applied_torque) < epsilon] = epsilon

    # Calculate the ratio
    force_ratio = projected_joint_forces / applied_torque  # Element-wise division

    return force_ratio


def get_dof_weight_distribution(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int] | torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Calculate weight distribution across forelegs and hindlegs based on joint forces, considering the gravity component.

    Args:
        env: The simulation environment.
        env_ids: Environment IDs (optional, not used here).
        asset_cfg: Asset configuration.

    Returns:
        weight_distribution: A tensor with weight distribution across forelegs and hindlegs.
                             Shape: (num_envs, 2) -> [weight_on_forelegs, weight_on_hindlegs]
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = slice(None)
    force_from_child_link_to_joints = asset.root_physx_view.get_link_incoming_joint_force().to(env.device)[env_ids]
    weight_forces = force_from_child_link_to_joints[..., 2]
    return weight_forces
