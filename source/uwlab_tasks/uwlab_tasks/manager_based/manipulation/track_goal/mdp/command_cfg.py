# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from .command import HandJointCommand


@configclass
class HandJointCommandCfg(CommandTermCfg):
    """Configuration for uniform pose command generator."""

    class_type: type = HandJointCommand

    asset_cfg: SceneEntityCfg = MISSING  # type: ignore

    articulation_vis_cfg: SceneEntityCfg = MISSING  # type: ignore

    predefined_hand_joint_goals: list[list[float]] = MISSING  # type: ignore

    wrist_pose_term: str = MISSING  # type: ignore
