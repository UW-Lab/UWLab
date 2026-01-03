# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass

from .articulation_drive import ArticulationDrive


@configclass
class ArticulationDriveCfg:
    """Configuration parameters for an articulation view."""

    class_type: Callable[..., ArticulationDrive] = MISSING  # type: ignore

    use_multiprocessing: bool = False

    dt = 0.01

    device: str = "cpu"
