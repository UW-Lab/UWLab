# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

##
# Configuration
##

"""
XARM_LEAP
"""
# fmt: off
XARM_LEAP_DEFAULT_JOINT_POS = {".*": 0.0}
# fmt: on

XARM_LEAP_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UFactory/Xarm5LeapHand/leap_xarm_ikpoints.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=1, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0, 0, 0), rot=(1, 0, 0, 0), joint_pos=XARM_LEAP_DEFAULT_JOINT_POS),
    soft_joint_pos_limit_factor=1,
)

IMPLICIT_XARM_LEAP = XARM_LEAP_ARTICULATION.copy()  # type: ignore
IMPLICIT_XARM_LEAP.actuators = {
    "arm1": ImplicitActuatorCfg(
        joint_names_expr=["joint.*"],
        stiffness={"joint[1-2]": 1000, "joint3": 800, "joint[4-5]": 600},
        damping=100.0,
        # velocity_limit=3.14,
        effort_limit={"joint[1-2]": 50, "joint3": 30, "joint[4-5]": 20},
    ),
    "j": ImplicitActuatorCfg(
        joint_names_expr=["j[0-9]+"],
        stiffness=20.0,
        damping=1.0,
        armature=0.001,
        friction=0.2,
        # velocity_limit=8.48,
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
