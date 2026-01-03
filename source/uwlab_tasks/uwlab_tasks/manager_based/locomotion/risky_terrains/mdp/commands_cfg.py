# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass

from .commands import UniformPolarPose2dCommand


@configclass
class UniformPolarPose2dCommandCfg(CommandTermCfg):
    class_type: type = UniformPolarPose2dCommand

    asset_name: str = MISSING

    simple_heading: bool = MISSING

    @configclass
    class Ranges:
        distance_range: tuple[float, float] = MISSING
        heading: tuple[float, float] = MISSING

    ranges: Ranges = MISSING

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/pose_goal"
    )
    current_pose_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="Visuals/Command/current_pose"
    )

    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.2)
    current_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.2)
