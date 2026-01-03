# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions import BinaryJointPositionActionCfg, DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from ...factory_tasks import FRANKA_PANDA_CFG
from ...gearmesh_env_cfg import GearMeshEnvCfg
from ...nutthread_env_cfg import NutThreadEnvCfg
from ...peginsert_env_cfg import PegInsertEnvCfg
from . import joint_pos_env_cfg


@configclass
class FrankaFactoryIkDelEnvMixIn:
    def __post_init__(self: joint_pos_env_cfg.FrankaNutThreadEnvCfg):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["panda_arm1"].stiffness = 800
        self.scene.robot.actuators["panda_arm2"].stiffness = 700
        self.scene.robot.actuators["panda_arm1"].damping = 30
        self.scene.robot.actuators["panda_arm2"].damping = 30
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=(0.0, 0.0, 0.107)),
        )
        self.actions.gripper_action = BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )


@configclass
class FrankaNutThreadIkDeltaEnvCfg(FrankaFactoryIkDelEnvMixIn, NutThreadEnvCfg):
    pass


@configclass
class FrankaGearMeshIkDeltaEnvCfg(FrankaFactoryIkDelEnvMixIn, GearMeshEnvCfg):
    pass


@configclass
class FrankaPegInsertIkDeltaEnvCfg(FrankaFactoryIkDelEnvMixIn, PegInsertEnvCfg):
    pass
