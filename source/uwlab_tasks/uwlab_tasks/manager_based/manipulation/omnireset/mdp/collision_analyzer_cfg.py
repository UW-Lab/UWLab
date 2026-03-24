# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .collision_analyzer import CollisionAnalyzer


@configclass
class CollisionAnalyzerCfg:

    class_type: type[CollisionAnalyzer] = CollisionAnalyzer

    num_points: int = 32

    max_dist: float = 0.5

    min_dist: float = 0.0

    asset_cfg: SceneEntityCfg = MISSING

    obstacle_cfgs: list[SceneEntityCfg] = MISSING
