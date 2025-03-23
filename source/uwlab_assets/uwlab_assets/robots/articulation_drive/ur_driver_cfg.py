# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

from isaaclab.utils import configclass
from uwlab.assets.articulation.articulation_drive import ArticulationDriveCfg

from .ur_driver import URDriver


@configclass
class URDriverCfg(ArticulationDriveCfg):
    class_type: Callable[..., URDriver] = URDriver

    ip: str = MISSING  # type: ignore

    port: int = MISSING  # type: ignore
