# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg, JointPositionActionCfg
from isaaclab.utils import configclass

from uwlab.controllers.differential_ik_cfg import MultiConstraintDifferentialIKControllerCfg
from uwlab.envs.mdp.actions.actions_cfg import MultiConstraintsDifferentialInverseKinematicsActionCfg

"""
LEAP ACTIONS
"""

LEAP_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot", joint_names=["w.*", "j.*"], scale=0.1
)

LEAP_JOINT_EFFORT: JointEffortActionCfg = JointEffortActionCfg(
    asset_name="robot", joint_names=["w.*", "j.*"], scale=0.1
)

LEAP_MC_IKABSOLUTE = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["w.*", "j.*"],
    body_name=["wrist", "pip", "pip_2", "pip_3", "thumb_fingertip", "tip", "tip_2", "tip_3"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
)

LEAP_MC_IKDELTA = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["w.*", "j.*"],
    body_name=["wrist", "pip", "pip_2", "pip_3", "thumb_fingertip", "tip", "tip_2", "tip_3"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="position", use_relative_mode=True, ik_method="dls"
    ),
    scale=0.1,
)


@configclass
class LeapMcIkAbsoluteAction:
    joint_pos = LEAP_MC_IKABSOLUTE


@configclass
class LeapMcIkDeltaAction:
    joint_pos = LEAP_MC_IKDELTA


@configclass
class LeapJointPositionAction:
    joint_pos = LEAP_JOINT_POSITION


@configclass
class LeapJointEffortAction:
    joint_pos = LEAP_JOINT_EFFORT
