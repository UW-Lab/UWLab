# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.utils import configclass

from .differential_ik import MultiConstraintDifferentialIKController


@configclass
class MultiConstraintDifferentialIKControllerCfg(DifferentialIKControllerCfg):
    """Configuration for multi-constraint differential inverse kinematics controller."""

    class_type: type = MultiConstraintDifferentialIKController
    """The associated controller class."""
