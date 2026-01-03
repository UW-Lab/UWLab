# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

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
