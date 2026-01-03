# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import uwlab_tasks.manager_based.manipulation.factory_extension.mdp as mdp

from ...factory_assets_cfg import FRANKA_PANDA_CFG
from ...factory_env_base import FactoryBaseEnvCfg
from ...gearmesh_env_cfg import GearMeshEnvCfg
from ...nutthread_env_cfg import NutThreadEnvCfg
from ...peginsert_env_cfg import PegInsertEnvCfg


@configclass
class ActionCfg:
    arm_action = mdp.RelativeJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        scale=0.02,
        use_zero_offset=True,
    )

    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.0},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class FrankaFactoryEnvMixIn:
    actions: ActionCfg = ActionCfg()

    def __post_init__(self: FactoryBaseEnvCfg):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["panda_arm1"].stiffness = 80.0
        self.scene.robot.actuators["panda_arm1"].damping = 4.0
        self.scene.robot.actuators["panda_arm2"].stiffness = 80.0
        self.scene.robot.actuators["panda_arm2"].damping = 4.0


@configclass
class FrankaNutThreadEnvCfg(FrankaFactoryEnvMixIn, NutThreadEnvCfg):
    pass


@configclass
class FrankaGearMeshEnvCfg(FrankaFactoryEnvMixIn, GearMeshEnvCfg):
    pass


@configclass
class FrankaPegInsertEnvCfg(FrankaFactoryEnvMixIn, PegInsertEnvCfg):
    pass
