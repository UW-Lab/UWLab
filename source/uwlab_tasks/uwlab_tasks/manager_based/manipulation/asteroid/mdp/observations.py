# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation terms for the ASTEROID tactile / proprioceptive student policies."""

from __future__ import annotations

import math
import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def gripper_pos_normalized(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["left_inner_finger_knuckle_joint"]),
    full_close_angle: float = -math.pi / 4,
    scale_event_name: str | None = None,
    jitter_std: float = 0.0,
) -> torch.Tensor:
    """Robotiq 2F-85 gripper position as a ``[0, 1]`` scalar (0 = open, 1 = closed).

    Mirrors the real-world Robotiq POS register (``robotiq_gripper.get_current_position``),
    normalized to ``[0, 1]`` instead of the firmware's 0-255.

    Inverts the sim-side mapping ``inner_finger_knuckle_joint_angle = full_close_angle * pos``,
    where ``full_close_angle`` is the joint angle at fully-closed (negative on the UWLab URDF:
    ~-pi/4). For multi-joint ``asset_cfg`` the mean angle is used; the 2F-85's left/right
    knuckles are mimic-coupled so they track each other.

    Args:
        env: The environment.
        asset_cfg: Scene entity + joint(s) to read. Defaults to the left inner_finger_knuckle joint.
        full_close_angle: Joint angle at fully closed (radians, signed). Default ``-pi/4`` matches
            UWLab's 2F-85 URDF convention; pass ``+pi/4`` (or use a different joint) if your robot's
            joint axis is the opposite sign.
        scale_event_name: Name of a :class:`~.events.randomize_gripper_pos_affine` event term whose
            per-env ``scale`` / ``offset`` are applied to the reading (calibration-drift DR).
        jitter_std: Std of per-step Gaussian noise added on top (freshly sampled each call).

    Returns:
        Tensor of shape ``(num_envs, 1)`` in ``[0.0, 1.0]`` (before scale / offset / jitter).
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    angle = robot.data.joint_pos[:, joint_ids]
    if angle.dim() > 1 and angle.shape[-1] > 1:
        angle = angle.mean(dim=-1, keepdim=True)
    elif angle.dim() == 1:
        angle = angle.unsqueeze(-1)
    pos = (angle / full_close_angle).clamp(0.0, 1.0)
    if scale_event_name is not None:
        # For class-based terms (ManagerTermBase subclasses) cfg.func is the instantiated object.
        try:
            scale_term = env.event_manager.get_term_cfg(scale_event_name).func
        except ValueError as e:
            raise RuntimeError(
                f"gripper_pos_normalized: event term '{scale_event_name}' not registered on event_manager."
            ) from e
        if not hasattr(scale_term, "scale"):
            raise RuntimeError(
                f"gripper_pos_normalized: event term '{scale_event_name}' has no .scale attr "
                f"(got {type(scale_term).__name__})."
            )
        # scale / offset: (num_envs,) -> (num_envs, 1) for broadcast.
        pos = pos * scale_term.scale.unsqueeze(-1)
        if hasattr(scale_term, "offset"):
            pos = pos + scale_term.offset.unsqueeze(-1)
    if jitter_std > 0.0:
        pos = pos + torch.randn_like(pos) * jitter_std
    return pos.to(torch.float32)


def fingertip_contact_force_b(
    env: ManagerBasedRLEnv,
    contact_sensor_name: str,
    root_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    root_body_name: str = "robotiq_base_link",
) -> torch.Tensor:
    """Contact force from a single fingertip contact sensor, expressed in a body frame.

    Args:
        env: The environment to extract contact forces from.
        contact_sensor_name: Name of the contact sensor to read from.
        root_asset_cfg: Asset whose body frame the force is expressed in.
        root_body_name: Body of ``root_asset_cfg`` to use as reference frame.

    Returns:
        Contact force in body frame. Shape: ``(num_envs, 3)``.
    """
    root_asset: Articulation = env.scene[root_asset_cfg.name]
    root_body_idx = root_asset.body_names.index(root_body_name)
    root_quat_w = root_asset.data.body_link_quat_w[:, root_body_idx].view(-1, 4)

    contact_sensor = env.scene.sensors[contact_sensor_name]
    # force_matrix_w is flattened, so we reshape to (num_envs, 3)
    force_w = contact_sensor.data.force_matrix_w.view(env.num_envs, 3)

    # Rotation only: forces are free vectors.
    return math_utils.quat_apply_inverse(root_quat_w, force_w)
