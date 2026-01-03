# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialIKControllerCfg,
    DifferentialInverseKinematicsActionCfg,
    JointEffortActionCfg,
    JointPositionActionCfg,
)
from isaaclab.utils import configclass

"""
XARM GRIPPER ACTIONS
"""
XARM_UF_GRIPPER_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["joint.*", "drive_joint"],
    scale=1.0,
    use_default_offset=False,
)


XARM_UF_GRIPPER_JOINT_EFFORT: JointEffortActionCfg = JointEffortActionCfg(
    asset_name="robot", joint_names=["joint.*", "drive_joint"], scale=0.1
)


XARM_UF_GRIPPER_MC_IKABSOLUTE_ARM = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["joint.*"],
    body_name="link_tcp",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    scale=1,
)

XARM_UF_GRIPPER_MC_IKDELTA_ARM = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["joint.*"],
    body_name="link_tcp",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    scale=0.5,
)


XARM_GRIPPER_BINARY_ACTIONS = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["drive_joint"],
    open_command_expr={"drive_joint": 0.0},
    close_command_expr={"drive_joint": 1.0},
)


@configclass
class XarmUfGripperIkAbsoluteAction:
    joint_pos = XARM_UF_GRIPPER_MC_IKABSOLUTE_ARM
    gripper = XARM_GRIPPER_BINARY_ACTIONS


@configclass
class XarmUfGripperMcIkDeltaAction:
    joint_pos = XARM_UF_GRIPPER_MC_IKDELTA_ARM
    gripper = XARM_GRIPPER_BINARY_ACTIONS


@configclass
class XarmUfGripperJointPositionAction:
    joint_pos = XARM_UF_GRIPPER_JOINT_POSITION
    gripper = XARM_GRIPPER_BINARY_ACTIONS
