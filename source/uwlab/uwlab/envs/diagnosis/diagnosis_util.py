# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import isaaclab.utils.math as math_utils


@torch.jit.script
def get_dof_stress_von_mises_optimized(
    forces_w: torch.Tensor,
    torques_w: torch.Tensor,
    bodies_quat: torch.Tensor,
    A: torch.Tensor,
    I: torch.Tensor,
    J: torch.Tensor,
    c: float,
) -> torch.Tensor:
    # Transform forces and torques to body frame
    forces_b = math_utils.quat_apply_inverse(bodies_quat, forces_w)
    torques_b = math_utils.quat_apply_inverse(bodies_quat, torques_w)

    # Extract force and torque components
    F = forces_b  # Shape: (..., 3)
    M = torques_b  # Shape: (..., 3)

    # Precompute ratios
    F_over_A = F / A  # Shape: (..., 3)
    M_c_over_IJ = M * c  # Shape: (..., 3)
    M_c_over_IJ[..., :2] /= I  # M_x and M_y over I
    M_c_over_IJ[..., 2] /= J  # M_z over J

    # Compute axial stress
    sigma_axial = F_over_A[..., 2]  # F_z / A

    # Compute bending stress
    sigma_bending = M_c_over_IJ[..., 1] + M_c_over_IJ[..., 0]  # M_y*c/I + M_x*c/I

    # Compute shear stress due to torsion
    tau_torsion = M_c_over_IJ[..., 2]  # M_z*c/J

    # Compute shear stress due to shear forces
    tau_shear = torch.sqrt(F_over_A[..., 0] ** 2 + F_over_A[..., 1] ** 2)

    # Total equivalent shear stress
    tau_eq = torch.sqrt(tau_torsion**2 + tau_shear**2)

    # Total equivalent normal stress
    sigma_eq = sigma_axial + sigma_bending

    # Compute von Mises stress
    sigma_von_mises = torch.sqrt(sigma_eq**2 + 3.0 * tau_eq**2)

    return sigma_von_mises
