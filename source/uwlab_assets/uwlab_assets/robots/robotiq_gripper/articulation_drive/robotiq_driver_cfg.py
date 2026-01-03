# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable

from isaaclab.utils import configclass

from uwlab.assets.articulation.articulation_drive import ArticulationDriveCfg

from .robotiq_driver import RobotiqDriver


@configclass
class RobotiqDriverCfg(ArticulationDriveCfg):
    class_type: Callable[..., RobotiqDriver] = RobotiqDriver
