# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# ignore private usage of cache helpers in unit tests
# pyright: reportPrivateUsage=none

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "uwlab_assets" / "robots" / "ur5e_robotiq_gripper" / "kinematics.py"
)


def _load_kinematics_module(monkeypatch):
    """Load the kinematics module with a minimal isaaclab stub for unit testing."""
    isaaclab_module = types.ModuleType("isaaclab")
    utils_module = types.ModuleType("isaaclab.utils")
    assets_module = types.ModuleType("isaaclab.utils.assets")
    assets_module.retrieve_file_path = lambda path, download_dir=None: path
    utils_module.assets = assets_module
    isaaclab_module.utils = utils_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.assets", assets_module)

    module_name = "uwlab_test_ur5e_kinematics"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _make_calibration(dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:
    return {
        "joints_xyz": torch.tensor(
            [
                [0.0, 0.0, 0.1625],
                [-0.4250, 0.0, 0.0],
                [-0.3922, 0.0, 0.0],
                [0.0, -0.1333, 0.1333],
                [0.0, 0.0997, 0.0996],
                [0.0, -0.0996, 0.0997],
            ],
            dtype=dtype,
        ),
        "joints_rpy": torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.5708, 0.0],
                [0.0, 0.0, 0.0],
                [1.5708, 0.0, 0.0],
                [-1.5708, 0.0, 0.0],
                [1.5708, 0.0, 0.0],
            ],
            dtype=dtype,
        ),
        "link_masses": torch.ones(6, dtype=dtype),
        "link_coms": torch.zeros(6, 3, dtype=dtype),
        "link_inertias": torch.zeros(6, 6, dtype=dtype),
    }


def _reference_compute_jacobian(
    module, joint_angles: torch.Tensor, calibration: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Reference implementation copied from the pre-optimization Jacobian path."""
    device = joint_angles.device
    dtype = joint_angles.dtype
    batch_size = joint_angles.shape[0]
    xyz_all = calibration["joints_xyz"].to(device=device, dtype=dtype)
    rpy_all = calibration["joints_rpy"].to(device=device, dtype=dtype)
    base_rotation = module.R_180Z.to(device=device, dtype=dtype)

    transform = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    for joint_idx in range(module.NUM_ARM_JOINTS):
        fixed_rot = module.rpy_to_matrix_torch(rpy_all[joint_idx])
        fixed_tf = torch.eye(4, device=device, dtype=dtype)
        fixed_tf[:3, :3] = fixed_rot
        fixed_tf[:3, 3] = xyz_all[joint_idx]
        fixed_tf = fixed_tf.unsqueeze(0).repeat(batch_size, 1, 1)
        theta = joint_angles[:, joint_idx]
        ct, st = torch.cos(theta), torch.sin(theta)
        joint_tf = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        joint_tf[:, 0, 0] = ct
        joint_tf[:, 0, 1] = -st
        joint_tf[:, 1, 0] = st
        joint_tf[:, 1, 1] = ct
        transform = torch.bmm(torch.bmm(transform, fixed_tf), joint_tf)
    ee_pos = transform[:, :3, 3]

    jacobian = torch.zeros(batch_size, 6, module.NUM_ARM_JOINTS, device=device, dtype=dtype)
    transform = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
    for joint_idx in range(module.NUM_ARM_JOINTS):
        fixed_rot = module.rpy_to_matrix_torch(rpy_all[joint_idx])
        fixed_tf = torch.eye(4, device=device, dtype=dtype)
        fixed_tf[:3, :3] = fixed_rot
        fixed_tf[:3, 3] = xyz_all[joint_idx]
        fixed_tf = fixed_tf.unsqueeze(0).repeat(batch_size, 1, 1)
        joint_frame_tf = torch.bmm(transform, fixed_tf)
        joint_axis = joint_frame_tf[:, :3, 2]
        joint_pos = joint_frame_tf[:, :3, 3]
        jacobian[:, :3, joint_idx] = torch.cross(joint_axis, ee_pos - joint_pos, dim=1)
        jacobian[:, 3:, joint_idx] = joint_axis
        theta = joint_angles[:, joint_idx]
        ct, st = torch.cos(theta), torch.sin(theta)
        joint_rot_tf = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
        joint_rot_tf[:, 0, 0] = ct
        joint_rot_tf[:, 0, 1] = -st
        joint_rot_tf[:, 1, 0] = st
        joint_rot_tf[:, 1, 1] = ct
        transform = torch.bmm(joint_frame_tf, joint_rot_tf)

    base_rotation = base_rotation.unsqueeze(0).repeat(batch_size, 1, 1)
    jacobian[:, :3, :] = torch.bmm(base_rotation, jacobian[:, :3, :])
    jacobian[:, 3:, :] = torch.bmm(base_rotation, jacobian[:, 3:, :])
    return jacobian


@pytest.fixture
def kinematics_module(monkeypatch):
    return _load_kinematics_module(monkeypatch)


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
    ],
)
def test_compute_jacobian_matches_reference(kinematics_module, monkeypatch, batch_size, device):
    calibration = _make_calibration()
    monkeypatch.setattr(kinematics_module, "_load_calibration", lambda: calibration)
    kinematics_module._get_calibrated_kinematics_cache.cache_clear()

    torch.manual_seed(7)
    joint_angles = torch.randn(batch_size, kinematics_module.NUM_ARM_JOINTS, device=device, dtype=torch.float32)

    fast_jacobian = kinematics_module.compute_jacobian_analytical(joint_angles)
    reference_jacobian = _reference_compute_jacobian(kinematics_module, joint_angles, calibration)

    assert torch.allclose(fast_jacobian, reference_jacobian, atol=1e-6, rtol=1e-5)


def test_compute_jacobian_reuses_cached_fixed_rotations(kinematics_module, monkeypatch):
    calibration = _make_calibration()
    monkeypatch.setattr(kinematics_module, "_load_calibration", lambda: calibration)
    kinematics_module._get_calibrated_kinematics_cache.cache_clear()

    original_rpy_to_matrix = kinematics_module.rpy_to_matrix_torch
    call_count = {"value": 0}

    def counting_rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
        call_count["value"] += 1
        return original_rpy_to_matrix(rpy)

    monkeypatch.setattr(kinematics_module, "rpy_to_matrix_torch", counting_rpy_to_matrix)

    joint_angles = torch.randn(8, kinematics_module.NUM_ARM_JOINTS, dtype=torch.float32)
    kinematics_module.compute_jacobian_analytical(joint_angles)
    kinematics_module.compute_jacobian_analytical(joint_angles.clone())

    assert call_count["value"] == 1
