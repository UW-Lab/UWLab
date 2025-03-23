# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from isaaclab.utils import configclass
from uwlab.assets.articulation.articulation_drive import ArticulationDriveCfg

from .xarm_driver import XarmDriver


@configclass
class XarmDriverCfg(ArticulationDriveCfg):
    class_type: Callable[..., XarmDriver] = XarmDriver

    work_space_limit: list[list[float]] = MISSING  # type: ignore

    ip: str = MISSING  # type: ignore

    is_radian: bool = True

    p_gain_scaler: float = 0.01
