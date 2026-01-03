# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    DifferentialInverseKinematicsActionCfg,
    JointPositionActionCfg,
)
from isaaclab.utils import configclass

JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*", "panda_finger_joint.*"],
    scale=0.1,
)

JOINT_IKDELTA: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_hand",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    scale=0.5,
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034), rot=(1.0, 0.0, 0, 0)),
)

JOINT_IKABSOLUTE: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["panda_joint.*"],
    body_name="panda_hand",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    scale=1,
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.1034), rot=(1.0, 0.0, 0, 0)),
)

BINARY_GRIPPER = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["panda_finger.*"],
    open_command_expr={"panda_finger_.*": 0.04},
    close_command_expr={"panda_finger_.*": 0.0},
)


@configclass
class JointPositionAction:
    jointpos = JOINT_POSITION


@configclass
class IkDeltaAction:
    body_joint_pos = JOINT_IKDELTA
    gripper_joint_pos = BINARY_GRIPPER


@configclass
class IkAbsoluteAction:
    body_joint_pos = JOINT_IKABSOLUTE
    gripper_joint_pos = BINARY_GRIPPER
