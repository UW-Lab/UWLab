# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from typing import List, TypedDict


class ArticulationDriveData(TypedDict):
    is_running: bool
    close: bool
    link_names: List[str]
    dof_names: List[str]
    dof_types: List[str]
    pos: torch.Tensor
    vel: torch.Tensor
    torque: torch.Tensor
    pos_target: torch.Tensor
    vel_target: torch.Tensor
    eff_target: torch.Tensor
    link_transforms: torch.Tensor
    link_velocities: torch.Tensor
    link_mass: torch.Tensor
    link_inertia: torch.Tensor
    link_coms: torch.Tensor
    mass_matrix: torch.Tensor
    dof_stiffness: torch.Tensor
    dof_armatures: torch.Tensor
    dof_frictions: torch.Tensor
    dof_damping: torch.Tensor
    dof_limits: torch.Tensor
    dof_max_forces: torch.Tensor
    dof_max_velocity: torch.Tensor
    jacobians: torch.Tensor
