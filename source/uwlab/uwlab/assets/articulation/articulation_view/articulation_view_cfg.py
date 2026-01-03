# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from .articulation_view import ArticulationView
from .bullet_articulation_view import BulletArticulationView

if TYPE_CHECKING:
    from ..articulation_drive import ArticulationDriveCfg


@configclass
class ArticulationViewCfg:
    """Configuration parameters for an articulation view."""

    class_type: Callable[[ArticulationViewCfg, str], ArticulationView] = MISSING  # type: ignore

    device: str = "cpu"


@configclass
class BulletArticulationViewCfg(ArticulationViewCfg):
    class_type: Callable[..., BulletArticulationView] = BulletArticulationView

    drive_cfg: ArticulationDriveCfg = MISSING  # type: ignore

    urdf: str = MISSING  # type: ignore

    debug_visualize: bool = False

    use_multiprocessing: bool = False

    isaac_joint_names: list[str] = MISSING  # type: ignore
    """Joint names in the Isaac Sim order, this is important when real is executing the action target from isaac sim during
    sync mode. If the real environment are run independently, this field is not necessary."""

    dummy_mode: bool = False

    dt: float = 0.02
