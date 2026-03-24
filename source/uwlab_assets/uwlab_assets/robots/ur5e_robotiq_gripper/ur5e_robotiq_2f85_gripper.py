# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the UR5e + Robotiq 2F-85 robot.

The following configurations are available:

* :obj:`UR5E_ARTICULATION`: Base articulation (USD, init state).
* :obj:`EXPLICIT_UR5E_ROBOTIQ_2F85`: Full robot with DelayedPDActuator arm (PD delay, for sim2real finetuning).
* :obj:`IMPLICIT_UR5E_ROBOTIQ_2F85`: Full robot with ImplicitActuator arm (no motor delay, for RL training).
* :obj:`UR5E_ROBOTIQ_2F85`: Alias for ``EXPLICIT_UR5E_ROBOTIQ_2F85`` (backward compatibility).
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

ROBOTIQ_2F85_DEFAULT_JOINT_POS = {
    "finger_joint": 0.0,
    "right_outer_knuckle_joint": 0.0,
    "left_inner_knuckle_joint": 0.0,
    "right_inner_knuckle_joint": 0.0,
    "left_inner_finger_knuckle_joint": 0.0,
    "right_inner_finger_knuckle_joint": 0.0,
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

UR5E_VELOCITY_LIMITS = {
    "shoulder_pan_joint": 1.5708,
    "shoulder_lift_joint": 1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": 3.1415,
    "wrist_2_joint": 3.1415,
    "wrist_3_joint": 3.1415,
}

UR5E_EFFORT_LIMITS = {
    "shoulder_pan_joint": 150.0,
    "shoulder_lift_joint": 150.0,
    "elbow_joint": 150.0,
    "wrist_1_joint": 28.0,
    "wrist_2_joint": 28.0,
    "wrist_3_joint": 28.0,
}

UR5E_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UniversalRobots/Ur5e2f85RobotiqGripperCalibrated/ur5e_robotiq_gripper_d415_mount_safety_calibrated.usd",
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
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UniversalRobots/2f85RobotiqGripperCalibrated/robotiq_2f85_gripper_calibrated.usd",
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
            effort_limit_sim=60,
        ),
    },
    soft_joint_pos_limit_factor=1,
)

EXPLICIT_UR5E_ROBOTIQ_2F85 = UR5E_ARTICULATION.copy()  # type: ignore
EXPLICIT_UR5E_ROBOTIQ_2F85.actuators = {
    "arm": DelayedPDActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit=UR5E_EFFORT_LIMITS,
        effort_limit_sim=UR5E_EFFORT_LIMITS,
        velocity_limit=UR5E_VELOCITY_LIMITS,
        velocity_limit_sim=UR5E_VELOCITY_LIMITS,
        min_delay=0,
        max_delay=1,
    ),
    "gripper": ROBOTIQ_2F85.actuators["gripper"],
}

IMPLICIT_UR5E_ROBOTIQ_2F85 = UR5E_ARTICULATION.copy()  # type: ignore
IMPLICIT_UR5E_ROBOTIQ_2F85.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["shoulder.*", "elbow.*", "wrist.*"],
        stiffness=0.0,
        damping=0.0,
        effort_limit_sim=UR5E_EFFORT_LIMITS,
        velocity_limit_sim=UR5E_VELOCITY_LIMITS,
    ),
    "gripper": ROBOTIQ_2F85.actuators["gripper"],
}
