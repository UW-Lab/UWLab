# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_tasks.manager_based.manipulation.omnireset.mdp.commands_cfg import TaskDependentCommandCfg

from .commands import PickTaskCommand


@configclass
class PickTaskCommandCfg(TaskDependentCommandCfg):
    """Configuration for :class:`~.commands.PickTaskCommand`."""

    class_type: type = PickTaskCommand

    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")

    insertive_asset_cfg: SceneEntityCfg = MISSING

    success: str | None = "env.reward_manager.get_term_cfg('progress_context').func.success"
    """Expression (evaluated with ``env`` bound) yielding a per-env success mask used for the
    end-of-episode success-rate metric. Set to ``None`` to disable."""
