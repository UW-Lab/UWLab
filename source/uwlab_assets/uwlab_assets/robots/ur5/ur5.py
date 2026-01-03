# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR5 robots.

The following configurations are available:

* :obj:`UR5_CFG`: Ur5 robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

UR5_DEFAULT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": 4.7112,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": -1.5708,
    "finger_joint": 0.0,
    "right_outer.*": 0.0,
    "left_outer.*": 0.0,
    "left_inner_finger_knuckle_joint": 0.0,
    "right_inner_finger_knuckle_joint": 0.0,
    "left_inner_finger_joint": -0.785398,
    "right_inner_finger_joint": 0.785398,
}

UR5_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UniversalRobots/Ur5RobotiqGripper/ur5_robotiq_gripper_backup.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(0, 0, 0, 1), joint_pos=UR5_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

IMPLICIT_UR5 = UR5_ARTICULATION.copy()  # type: ignore
IMPLICIT_UR5.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=261.8,
        damping=26.18,
        # velocity_limit=3.14,
        effort_limit={"shoulder.*": 9000, "elbow.*": 9000, "wrist.*": 1680},
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        stiffness=17,
        damping=5,
        # velocity_limit=2.27,
        effort_limit=165,
    ),
    "inner_finger": ImplicitActuatorCfg(
        joint_names_expr=[".*_inner_finger_joint"],
        stiffness=0.2,
        damping=0.02,
        # velocity_limit=5.3,
        effort_limit=0.5,
    ),
}
