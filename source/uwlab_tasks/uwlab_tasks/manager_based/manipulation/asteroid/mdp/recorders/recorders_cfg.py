# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from uwlab_tasks.manager_based.manipulation.omnireset.mdp.recorders.recorders_cfg import (
    ActionStateRecorderManagerCfg,
)

from . import recorders


@configclass
class PreStepExpertMaskRecorderCfg(RecorderTermCfg):
    """Configuration for the expert action mask recorder term (for DAgger-style data)."""

    class_type: type[RecorderTerm] = recorders.PreStepExpertMaskRecorder


@configclass
class AsteroidActionStateRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """OmniReset's action/state recorder plus the per-step expert mask."""

    record_pre_step_expert_mask = PreStepExpertMaskRecorderCfg()
