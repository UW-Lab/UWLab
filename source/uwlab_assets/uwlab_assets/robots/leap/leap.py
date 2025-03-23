# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

##
# Configuration
##

# fmt: off
LEAP_DEFAULT_JOINT_POS = {".*": 0.0}

LEAP6D_DEFAULT_JOINT_POS = {".*": 0.0}
# fmt: on


LEAP_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/LeapHand/leap_hand.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=LEAP_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

LEAP6D_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/LeapHand/leap_hand_6d.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=1, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=LEAP6D_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

IMPLICIT_LEAP = LEAP_ARTICULATION.copy()
IMPLICIT_LEAP.actuators = {
    "j": ImplicitActuatorCfg(
        joint_names_expr=["j.*"],
        stiffness=200.0,
        damping=30.0,
        armature=0.001,
        friction=0.2,
        velocity_limit=8.48,
        effort_limit=0.95,
    ),
}

IMPLICIT_LEAP6D = LEAP6D_ARTICULATION.copy()  # type: ignore
IMPLICIT_LEAP6D.actuators = {
    "w": ImplicitActuatorCfg(
        joint_names_expr=["w.*"],
        stiffness=200.0,
        damping=50.0,
        armature=0.001,
        friction=0.2,
        velocity_limit=1,
        effort_limit=50,
    ),
    "j": ImplicitActuatorCfg(
        joint_names_expr=["j.*"],
        stiffness=200.0,
        damping=30.0,
        armature=0.001,
        friction=0.2,
        velocity_limit=8.48,
        effort_limit=0.95,
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
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/palm_lower",
            name="ee",
            offset=OffsetCfg(
                pos=(-0.028, -0.04, -0.07),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        ),
    ],
)
