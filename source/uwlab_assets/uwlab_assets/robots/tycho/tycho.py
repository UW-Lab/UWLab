# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Tycho robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


##
# Configuration
##

HEBI_DEFAULT_JOINTPOS = {
    "HEBI_base_X8_9": -2.2683857389667805,
    "HEBI_shoulder_X8_16": 1.5267610481188283,
    "HEBI_elbow_X8_9": 2.115358222505881,
    "HEBI_wrist1_X5_1": 0.5894993521468314,
    "HEBI_wrist2_X5_1": 0.8740650991816328,
    "HEBI_wrist3_X5_1": 0.0014332898815118368,
    "HEBI_chopstick_X5_1": -0.36,
}

HEBI_ARTICULATION = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/HebiRobotic/Tycho/tycho.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False, solver_position_iteration_count=32, solver_velocity_iteration_count=1
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(rot=(0.7071068, 0, 0, 0.7071068), joint_pos=HEBI_DEFAULT_JOINTPOS),
    soft_joint_pos_limit_factor=1,
)

HEBI_IMPLICIT_ACTUATOR_CFG = HEBI_ARTICULATION.copy()  # type: ignore
HEBI_IMPLICIT_ACTUATOR_CFG.actuators = {
    "arm": ImplicitActuatorCfg(
        joint_names_expr=["HEBI_.*"],
        stiffness={"HEBI_(base|elbow|shoulder).*": 120.0, "HEBI_(wrist|chopstick).*": 40.0},
        damping={"HEBI_(base|elbow|shoulder).*": 20.0, "HEBI_(wrist|chopstick).*": 3.0},
        effort_limit={"HEBI_(base|elbow).*": 23.3, "HEBI_shoulder.*": 44.7632, "HEBI_(wrist|chopstick).*": 2.66},
        # velocity_limit=1,
    ),
}


"""
FRAMES
"""
marker_cfg = FRAME_MARKER_CFG.copy()  # type: ignore
marker_cfg.markers["frame"].scale = (0.01, 0.01, 0.01)
marker_cfg.prim_path = "/Visuals/FrameTransformer"

FRAME_EE = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/static_chop_tip",
            name="ee",
            offset=OffsetCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        ),
    ],
)


FRAME_FIXED_CHOP_TIP = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist3_chopstick",
            name="fixed_chop_tip",
            offset=OffsetCfg(
                pos=(0.13018, 0.07598, 0.06429),
            ),
        ),
    ],
)

FRAME_FIXED_CHOP_END = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/wrist3_chopstick",
            name="fixed_chop_end",
            offset=OffsetCfg(
                pos=(-0.13134, 0.07598, 0.06424),
            ),
        ),
    ],
)

FRAME_FREE_CHOP_TIP = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/end_effector",
            name="free_chop_tip",
            offset=OffsetCfg(
                pos=(0.12001, 0.05445, 0.00229),
            ),
        ),
    ],
)

FRAME_FREE_CHOP_END = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base_shoulder",
    debug_vis=False,
    visualizer_cfg=marker_cfg,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/end_effector",
            name="free_chop_end",
            offset=OffsetCfg(
                pos=(-0.11378, -0.04546, 0.00231),
            ),
        ),
    ],
)
