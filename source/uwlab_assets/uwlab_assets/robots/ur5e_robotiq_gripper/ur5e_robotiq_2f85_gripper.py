# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR5 robots.

The following configurations are available:

* :obj:`UR5E_CFG`: Ur5e robot
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

ROBOTIQ_2F85_DEFAULT_JOINT_POS = {
    "finger_joint": 0.0,
    "right_outer.*": 0.0,
    "left_outer.*": 0.0,
    "left_inner_finger_knuckle_joint": 0.0,
    "right_inner_finger_knuckle_joint": 0.0,
    "left_inner_finger_joint": -0.785398,
    "right_inner_finger_joint": 0.785398,
}

UR5E_DEFAULT_JOINT_POS = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": -1.5708,
    **ROBOTIQ_2F85_DEFAULT_JOINT_POS,
}

UR5E_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UniversalRobots/Ur5e2f85RobotiqGripper/ur5e_robotiq_gripper_d415_mount_safety.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=UR5E_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

ROBOTIQ_2F85 = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/RobotiqGripper",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UniversalRobots/2f85RobotiqGripper/robotiq_2f85_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=36, solver_velocity_iteration_count=0
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0.1), rot=(1, 0, 0, 0), joint_pos=ROBOTIQ_2F85_DEFAULT_JOINT_POS
    ),
    actuators={
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            stiffness=17,
            damping=5,
            effort_limit_sim=165,
        ),
        "inner_finger": ImplicitActuatorCfg(
            joint_names_expr=[".*_inner_finger_joint"],
            stiffness=0.2,
            damping=0.02,
            effort_limit_sim=0.5,
        ),
    },
    soft_joint_pos_limit_factor=1,
)

IMPLICIT_UR5E_ROBOTIQ_2F85 = UR5E_ARTICULATION.copy()  # type: ignore
IMPLICIT_UR5E_ROBOTIQ_2F85.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness={
            "shoulder_pan_joint": 4.63,
            "shoulder_lift_joint": 5.41,
            "elbow_joint": 8.06,
            "wrist_1_joint": 7.28,
            "wrist_2_joint": 8.04,
            "wrist_3_joint": 7.18,
        },
        damping={
            "shoulder_pan_joint": 8.84,
            "shoulder_lift_joint": 6.47,
            "elbow_joint": 9.46,
            "wrist_1_joint": 2.80,
            "wrist_2_joint": 2.41,
            "wrist_3_joint": 1.90,
        },
        velocity_limit_sim=3.14,
        effort_limit_sim={
            "shoulder_pan_joint": 150.0,
            "shoulder_lift_joint": 150.0,
            "elbow_joint": 150.0,
            "wrist_1_joint": 28.0,
            "wrist_2_joint": 28.0,
            "wrist_3_joint": 28.0,
        },
        armature=0.01,
    ),
    "gripper": ROBOTIQ_2F85.actuators["gripper"],
    "inner_finger": ROBOTIQ_2F85.actuators["inner_finger"],
}

EXPLICIT_UR5E_ROBOTIQ_2F85 = UR5E_ARTICULATION.copy()  # type: ignore
EXPLICIT_UR5E_ROBOTIQ_2F85.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        velocity_limit_sim=3.14,
        effort_limit_sim={
            "shoulder_pan_joint": 150.0,
            "shoulder_lift_joint": 150.0,
            "elbow_joint": 150.0,
            "wrist_1_joint": 28.0,
            "wrist_2_joint": 28.0,
            "wrist_3_joint": 28.0,
        },
        armature=0.01,
    ),
    "gripper": ROBOTIQ_2F85.actuators["gripper"],
    "inner_finger": ROBOTIQ_2F85.actuators["inner_finger"],
}
