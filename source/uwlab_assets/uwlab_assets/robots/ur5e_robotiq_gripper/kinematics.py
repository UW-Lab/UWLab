# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Calibrated kinematics for UR5e.

Contains:
- Analytical FK, Jacobian, and Mass Matrix computation (batched PyTorch)

Calibrated joint parameters and link inertials are loaded lazily from
``metadata.yaml`` co-located with the robot USD (via :func:`_load_calibration`).

All functions operate on the 6 arm joints only and output in the REP-103
base_link frame (180 deg Z rotation from base_link_inertia).
"""

import functools
import os
import tempfile
from dataclasses import dataclass

import torch
import yaml

from isaaclab.utils.assets import retrieve_file_path

# ============================================================================
# Constants
# ============================================================================

ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
EE_BODY_NAME = "wrist_3_link"
NUM_ARM_JOINTS = 6

# 180 deg rotation around Z-axis (base_link_inertia -> base_link conversion)
R_180Z = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=torch.float32)


@dataclass(frozen=True)
class _CalibratedKinematicsCache:
    """Device-local calibrated constants reused across Jacobian evaluations."""

    joints_xyz: torch.Tensor
    fixed_rotations: torch.Tensor
    base_rotation: torch.Tensor


# ============================================================================
# Lazy-loaded calibration data (from metadata.yaml next to the robot USD)
# ============================================================================


@functools.lru_cache(maxsize=1)
def _load_calibration() -> dict[str, torch.Tensor]:
    """Download (once) and parse calibrated kinematics from the robot metadata."""
    from .ur5e_robotiq_2f85_gripper import UR5E_ARTICULATION

    usd_dir = os.path.dirname(UR5E_ARTICULATION.spawn.usd_path)
    meta_path = os.path.join(usd_dir, "metadata.yaml")
    local = retrieve_file_path(meta_path, download_dir=tempfile.gettempdir())
    with open(local) as f:
        metadata = yaml.safe_load(f)
    if metadata is None:
        raise RuntimeError(f"metadata.yaml is empty or failed to load: {local} (source: {meta_path})")
    joints = metadata["calibrated_joints"]
    inertials = metadata["link_inertials"]
    return {
        "joints_xyz": torch.tensor(joints["xyz"], dtype=torch.float32),
        "joints_rpy": torch.tensor(joints["rpy"], dtype=torch.float32),
        "link_masses": torch.tensor(inertials["masses"], dtype=torch.float32),
        "link_coms": torch.tensor(inertials["coms"], dtype=torch.float32),
        "link_inertias": torch.tensor(inertials["inertias"], dtype=torch.float32),
    }


# ============================================================================
# Kinematics helpers
# ============================================================================


def rpy_to_matrix_torch(rpy: torch.Tensor) -> torch.Tensor:
    """Convert roll-pitch-yaw to rotation matrix (single or batched)."""
    if rpy.dim() == 1:
        roll, pitch, yaw = rpy[0], rpy[1], rpy[2]
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        R = torch.stack([
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
            torch.stack([-sp, cp * sr, cp * cr]),
        ])
        return R
    else:
        roll, pitch, yaw = rpy[:, 0], rpy[:, 1], rpy[:, 2]
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        R = torch.zeros(rpy.shape[0], 3, 3, device=rpy.device, dtype=rpy.dtype)
        R[:, 0, 0] = cy * cp
        R[:, 0, 1] = cy * sp * sr - sy * cr
        R[:, 0, 2] = cy * sp * cr + sy * sr
        R[:, 1, 0] = sy * cp
        R[:, 1, 1] = sy * sp * sr + cy * cr
        R[:, 1, 2] = sy * sp * cr - cy * sr
        R[:, 2, 0] = -sp
        R[:, 2, 1] = cp * sr
        R[:, 2, 2] = cp * cr
        return R


def _resolve_kinematics_device(device: torch.device | str | None, fallback: torch.device) -> torch.device:
    """Resolve an optional explicit device override for analytical kinematics."""
    if device is None:
        return fallback
    return torch.device(device)


@functools.lru_cache(maxsize=None)
def _get_calibrated_kinematics_cache(
    device_type: str, device_index: int, dtype: torch.dtype
) -> _CalibratedKinematicsCache:
    """Materialize calibrated UR5e constants once per device/dtype."""
    resolved_device = torch.device(device_type if device_index < 0 else f"{device_type}:{device_index}")
    calibration = _load_calibration()
    joints_xyz = calibration["joints_xyz"].to(device=resolved_device, dtype=dtype)
    joints_rpy = calibration["joints_rpy"].to(device=resolved_device, dtype=dtype)
    return _CalibratedKinematicsCache(
        joints_xyz=joints_xyz,
        fixed_rotations=rpy_to_matrix_torch(joints_rpy),
        base_rotation=R_180Z.to(device=resolved_device, dtype=dtype),
    )


# ============================================================================
# Analytical Jacobian
# ============================================================================


def compute_jacobian_analytical(
    joint_angles: torch.Tensor, device: torch.device | str | None = None
) -> torch.Tensor:
    """Compute geometric Jacobian using calibrated kinematics (batched).

    Computes to wrist_3_link frame origin (NOT COM), matching real robot code.

    Args:
        joint_angles: (N, 6) joint angles in radians.
        device: Optional explicit device override. Defaults to ``joint_angles.device``.
    Returns:
        J: (N, 6, 6) Jacobian [linear; angular].
    """
    if joint_angles.ndim != 2 or joint_angles.shape[1] != NUM_ARM_JOINTS:
        raise ValueError(f"Expected joint_angles to have shape (N, {NUM_ARM_JOINTS}), got {tuple(joint_angles.shape)}")
    if not joint_angles.is_floating_point():
        raise TypeError("joint_angles must be a floating point tensor")

    resolved_device = _resolve_kinematics_device(device, joint_angles.device)
    joint_angles = joint_angles.to(device=resolved_device)
    batch_size = joint_angles.shape[0]
    device_index = resolved_device.index if resolved_device.index is not None else -1
    constants = _get_calibrated_kinematics_cache(resolved_device.type, device_index, joint_angles.dtype)

    # Use a compact recurrence over rotation/translation instead of rebuilding 4x4 transforms
    # at every step. This is the hot path used by RelCartesianOSCAction.
    rot_world = (
        torch.eye(3, device=resolved_device, dtype=joint_angles.dtype)
        .unsqueeze(0)
        .expand(batch_size, -1, -1)
        .clone()
    )
    pos_world = joint_angles.new_zeros(batch_size, 3)
    joint_axes = []
    joint_positions = []

    for joint_idx in range(NUM_ARM_JOINTS):
        joint_frame_rot = torch.matmul(rot_world, constants.fixed_rotations[joint_idx])
        joint_frame_pos = pos_world + torch.matmul(rot_world, constants.joints_xyz[joint_idx])
        joint_axes.append(joint_frame_rot[:, :, 2])
        joint_positions.append(joint_frame_pos)

        theta = joint_angles[:, joint_idx]
        ct = torch.cos(theta).unsqueeze(-1)
        st = torch.sin(theta).unsqueeze(-1)
        rot_world = torch.stack(
            (
                ct * joint_frame_rot[:, :, 0] + st * joint_frame_rot[:, :, 1],
                -st * joint_frame_rot[:, :, 0] + ct * joint_frame_rot[:, :, 1],
                joint_frame_rot[:, :, 2],
            ),
            dim=-1,
        )
        pos_world = joint_frame_pos

    joint_axes_tensor = torch.stack(joint_axes, dim=-1)
    joint_positions_tensor = torch.stack(joint_positions, dim=-1)
    ee_offsets = pos_world.unsqueeze(-1) - joint_positions_tensor
    linear_jacobian = torch.cross(joint_axes_tensor, ee_offsets, dim=1)

    base_rotation = constants.base_rotation.unsqueeze(0).expand(batch_size, -1, -1)
    jacobian = joint_angles.new_empty(batch_size, 6, NUM_ARM_JOINTS)
    jacobian[:, :3, :] = torch.bmm(base_rotation, linear_jacobian)
    jacobian[:, 3:, :] = torch.bmm(base_rotation, joint_axes_tensor)
    return jacobian


# ============================================================================
# Analytical Mass Matrix (CRBA)
# ============================================================================


def compute_mass_matrix_analytical(joint_angles: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Compute 6x6 joint-space mass matrix using CRBA.

    Uses the same inertia parameters as real robot for consistency.

    Args:
        joint_angles: (N, 6) joint angles in radians.
    Returns:
        M: (N, 6, 6) mass matrix.
    """
    N = joint_angles.shape[0]
    cal = _load_calibration()
    xyz_all = cal["joints_xyz"].to(device)
    rpy_all = cal["joints_rpy"].to(device)
    masses = cal["link_masses"].to(device)
    coms = cal["link_coms"].to(device)
    inertias = cal["link_inertias"].to(device)

    M = torch.zeros(N, 6, 6, device=device, dtype=torch.float32)

    R_fixed_all = []
    T_fixed_all = []
    for i in range(6):
        R_fixed = rpy_to_matrix_torch(rpy_all[i])
        T_fixed = torch.eye(4, device=device, dtype=torch.float32)
        T_fixed[:3, :3] = R_fixed
        T_fixed[:3, 3] = xyz_all[i]
        R_fixed_all.append(R_fixed)
        T_fixed_all.append(T_fixed.unsqueeze(0).expand(N, -1, -1).clone())

    transforms = []
    T = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
    transforms.append(T.clone())
    for i in range(6):
        theta = joint_angles[:, i]
        ct, st = torch.cos(theta), torch.sin(theta)
        T_joint = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
        T_joint[:, 0, 0] = ct
        T_joint[:, 0, 1] = -st
        T_joint[:, 1, 0] = st
        T_joint[:, 1, 1] = ct
        T = torch.bmm(torch.bmm(T, T_fixed_all[i]), T_joint)
        transforms.append(T.clone())

    def make_joint_rot(theta_batch):
        ct, st = torch.cos(theta_batch), torch.sin(theta_batch)
        T_rot = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
        T_rot[:, 0, 0] = ct
        T_rot[:, 0, 1] = -st
        T_rot[:, 1, 0] = st
        T_rot[:, 1, 1] = ct
        return T_rot

    for link_idx in range(6):
        m = masses[link_idx]
        com_local = coms[link_idx]
        I_local = inertias[link_idx]
        I_tensor = torch.zeros(3, 3, device=device, dtype=torch.float32)
        I_tensor[0, 0] = I_local[0]
        I_tensor[1, 1] = I_local[1]
        I_tensor[2, 2] = I_local[2]
        I_tensor[0, 1] = I_tensor[1, 0] = I_local[3]
        I_tensor[0, 2] = I_tensor[2, 0] = I_local[4]
        I_tensor[1, 2] = I_tensor[2, 1] = I_local[5]

        T_link = transforms[link_idx + 1]
        R_link = T_link[:, :3, :3]
        p_link = T_link[:, :3, 3]
        p_com = p_link + torch.bmm(R_link, com_local.view(1, 3, 1).expand(N, -1, -1)).squeeze(-1)
        I_tensor_batch = I_tensor.unsqueeze(0).expand(N, -1, -1)
        I_world = torch.bmm(torch.bmm(R_link, I_tensor_batch), R_link.transpose(-1, -2))

        T_j = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
        for j in range(link_idx + 1):
            T_joint_frame_j = torch.bmm(T_j, T_fixed_all[j])
            z_j = T_joint_frame_j[:, :3, 2]
            p_j = T_joint_frame_j[:, :3, 3]
            J_v_j = torch.cross(z_j, p_com - p_j, dim=1)
            J_w_j = z_j

            for k in range(j + 1):
                T_k = torch.eye(4, device=device, dtype=torch.float32).unsqueeze(0).repeat(N, 1, 1)
                for kk in range(k + 1):
                    T_joint_frame_kk = torch.bmm(T_k, T_fixed_all[kk])
                    if kk < k:
                        T_k = torch.bmm(T_joint_frame_kk, make_joint_rot(joint_angles[:, kk]))
                    else:
                        T_k = T_joint_frame_kk
                z_k = T_k[:, :3, 2]
                p_k = T_k[:, :3, 3]
                J_v_k = torch.cross(z_k, p_com - p_k, dim=1)
                J_w_k = z_k
                term1 = m * torch.sum(J_v_j * J_v_k, dim=1)
                term2 = torch.sum(J_w_j * torch.bmm(I_world, J_w_k.unsqueeze(-1)).squeeze(-1), dim=1)
                M[:, j, k] += term1 + term2
                if j != k:
                    M[:, k, j] += term1 + term2
            T_j = torch.bmm(T_joint_frame_j, make_joint_rot(joint_angles[:, j]))

    M += torch.eye(6, device=device, dtype=torch.float32).unsqueeze(0) * 1e-6
    return M
