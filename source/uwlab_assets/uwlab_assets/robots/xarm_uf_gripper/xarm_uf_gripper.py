# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Xarm 5 with UFactory gripper"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

##
# Configuration
##


# fmt: off
XARM_UF_GRIPPER_DEFAULT_JOINT_POS = {
    "drive_joint": 0.0, "joint1": 0.0, "joint2": 0.0, "joint3": -0.5, "joint4": 0.0, "joint5": 0.0,
}
# fmt: on

XARM_UF_GRIPPER_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UFactory/Xarm5UfGripper/xarm_gripper.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=1, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=XARM_UF_GRIPPER_DEFAULT_JOINT_POS
    ),
    soft_joint_pos_limit_factor=1,
)

IMPLICIT_XARM_UF_GRIPPER = XARM_UF_GRIPPER_ARTICULATION.copy()  # type: ignore
IMPLICIT_XARM_UF_GRIPPER.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["joint.*"],
        stiffness={"joint[1-2]": 500, "joint3": 500, "joint[4-5]": 400},
        damping=50.0,
        # velocity_limit=3.14,
        effort_limit={"joint[1-2]": 50, "joint3": 30, "joint[4-5]": 20},
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["drive_joint"],
        stiffness=20.0,
        damping=1.0,
        armature=0.001,
        friction=0.2,
        # velocity_limit=2,
        effort_limit=50,
    ),
}


"""
FRAMES
"""
marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
marker_cfg.prim_path = "/Visuals/FrameTransformer"

FRAME_EE = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/link_base",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(prim_path="{ENV_REGEX_NS}/Robot/link_tcp", name="ee"),
    ],
)
