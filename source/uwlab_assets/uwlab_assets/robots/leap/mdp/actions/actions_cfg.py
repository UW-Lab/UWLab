# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .actions import LeapJointPositionActionCorrection


@configclass
class LeapJointPositionActionCorrectionCfg(JointPositionActionCfg):
    """Configuration for inverse differential kinematics action term with multi constraints.
    This class amend attr body_name from type:str to type:list[str] reflecting its capability to
    received the desired positions, poses from multiple target bodies. This will be particularly
    useful for controlling dextrous hand robot with only positions of multiple key frame positions
    and poses, and output joint positions that satisfy key frame position/pose constrains

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    class_type: type[ActionTerm] = LeapJointPositionActionCorrection
