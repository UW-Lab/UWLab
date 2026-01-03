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
    joint_names=["HEBI_(base|elbow|shoulder|wrist|chopstick).*"],
    scale=0.1,
)


IKDELTA: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["HEBI_(base|elbow|shoulder|wrist|chopstick).*"],
    body_name="static_chop_tip",
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
    scale=0.05,
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0, 0)),
)


IKABSOLUTE: DifferentialInverseKinematicsActionCfg = DifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["HEBI_(base|shoulder|elbow|wrist).*"],
    body_name="static_chop_tip",  # Do not work if this is not end_effector
    controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
    scale=1,
    body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1, 0, 0, 0)),
)


BINARY_GRIPPER = BinaryJointPositionActionCfg(
    asset_name="robot",
    joint_names=["HEBI_chopstick_X5_1"],
    open_command_expr={"HEBI_chopstick_X5_1": -0.175},
    close_command_expr={"HEBI_chopstick_X5_1": -0.646},
)


@configclass
class IkdeltaAction:
    body_joint_pos = IKDELTA
    gripper_joint_pos = BINARY_GRIPPER


@configclass
class IkabsoluteAction:
    body_joint_pos = IKABSOLUTE
    gripper_joint_pos = BINARY_GRIPPER


@configclass
class JointPositionAction:
    jointpos = JOINT_POSITION
