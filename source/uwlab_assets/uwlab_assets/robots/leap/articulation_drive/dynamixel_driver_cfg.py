# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from isaaclab.utils import configclass
from uwlab.assets.articulation.articulation_drive import ArticulationDriveCfg

from .dynamixel_driver import DynamixelDriver


@configclass
class DynamixelDriverCfg(ArticulationDriveCfg):
    class_type: Callable[..., DynamixelDriver] = DynamixelDriver

    port: str = MISSING  # type: ignore

    hand_kI: float = 0.0  # type: ignore

    hand_curr_lim: float = 350  # type: ignore
