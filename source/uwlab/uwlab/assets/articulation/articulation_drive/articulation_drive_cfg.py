# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Callable

from isaaclab.utils import configclass

from .articulation_drive import ArticulationDrive


@configclass
class ArticulationDriveCfg:
    """Configuration parameters for an articulation view."""

    class_type: Callable[..., ArticulationDrive] = MISSING  # type: ignore

    use_multiprocessing: bool = False

    dt = 0.01

    device: str = "cpu"
