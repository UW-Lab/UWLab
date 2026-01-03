# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from uwlab.controllers.differential_ik_cfg import MultiConstraintDifferentialIKControllerCfg
from uwlab.envs.mdp.actions.actions_cfg import (
    MultiConstraintsDifferentialInverseKinematicsActionCfg,
    PCAJointPositionActionCfg,
)

"""
LEAP XARM ACTIONS
"""
XARM_LEAP_JOINT_POSITION: JointPositionActionCfg = JointPositionActionCfg(
    asset_name="robot",
    joint_names=["joint.*", "j[0-9]+"],
    scale=1.0,
    use_default_offset=False,
)


XARM_LEAP_MC_IKABSOLUTE = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["joint.*", "j[0-9]+"],
    body_name=["wrist", "pip", "pip_2", "pip_3", "tip", "thumb_tip", "tip_2", "tip_3"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
)

XARM_LEAP_MC_IKABSOLUTE_ARM = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["joint.*"],
    body_name=["wrist"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="pose", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
)


XARM_LEAP_MC_IKABSOLUTE_FINGER = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["j[0-9]+"],
    body_name=["pip", "pip_2", "pip_3", "tip", "tip_2", "tip_3", "thumb_tip"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls"
    ),
    scale=1,
)


XARM_LEAP_MC_IKDELTA = MultiConstraintsDifferentialInverseKinematicsActionCfg(
    asset_name="robot",
    joint_names=["joint.*", "j[0-9]+"],
    body_name=["wrist", "pip", "pip_2", "pip_3", "tip", "thumb_tip", "tip_2", "tip_3"],
    controller=MultiConstraintDifferentialIKControllerCfg(
        command_type="position", use_relative_mode=True, ik_method="dls"
    ),
    scale=1,
)


XARM_LEAP_PCA_JOINT_POSITION = PCAJointPositionActionCfg(
    asset_name="robot",
    joint_names=["joint.*", "j[0-9]+"],
    scale=1.0,
    eigenspace_path=f"{UWLAB_CLOUD_ASSETS_DIR}/dataset/misc/hammer_grasping_pca_components.npy",
    joint_range=(-3.14, 3.14),
)


@configclass
class XarmLeapSeparatedIkAbsoluteAction:
    joint_pos = XARM_LEAP_MC_IKABSOLUTE_ARM
    finger_pos = XARM_LEAP_MC_IKABSOLUTE_FINGER


@configclass
class XarmLeapMcIkAbsoluteAction:
    joint_pos = XARM_LEAP_MC_IKABSOLUTE


@configclass
class XarmLeapMcIkDeltaAction:
    joint_pos = XARM_LEAP_MC_IKDELTA


@configclass
class XarmLeapJointPositionAction:
    joint_pos = XARM_LEAP_JOINT_POSITION


@configclass
class XarmLeapPCAJointPositionAction:
    joint_pos = XARM_LEAP_PCA_JOINT_POSITION
