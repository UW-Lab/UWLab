# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .commands import TaskCommand, TaskDependentCommand


@configclass
class TaskDependentCommandCfg(CommandTermCfg):
    class_type: type = TaskDependentCommand

    reset_terms_when_resample: dict[str, EventTerm] = {}


@configclass
class TaskCommandCfg(TaskDependentCommandCfg):
    class_type: type = TaskCommand

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

    success_position_threshold: float = MISSING

    success_orientation_threshold: float = MISSING

    insertive_asset_cfg: SceneEntityCfg = MISSING

    receptive_asset_cfg: SceneEntityCfg = MISSING
