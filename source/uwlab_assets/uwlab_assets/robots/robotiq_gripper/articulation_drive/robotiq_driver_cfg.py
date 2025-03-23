# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Callable

from isaaclab.utils import configclass
from uwlab.assets.articulation.articulation_drive import ArticulationDriveCfg

from .robotiq_driver import RobotiqDriver


@configclass
class RobotiqDriverCfg(ArticulationDriveCfg):
    class_type: Callable[..., RobotiqDriver] = RobotiqDriver
