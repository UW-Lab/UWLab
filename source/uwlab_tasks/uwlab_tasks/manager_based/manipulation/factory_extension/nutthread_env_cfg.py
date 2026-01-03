# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .assembly_keypoints import KEYPOINTS_BOLTM16, KEYPOINTS_NUTM16
from .factory_env_base import FactoryBaseEnvCfg, FactoryEventCfg, FactoryObservationsCfg, FactoryRewardsCfg


@configclass
class NutThreadObservationsCfg(FactoryObservationsCfg):
    def __post_init__(self):
        # policy
        self.policy.end_effector_vel_lin_ang_b.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.end_effector_pose.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.fixed_asset_in_end_effector_frame.params["target_asset_cfg"] = SceneEntityCfg("bolt_m16")
        self.policy.fixed_asset_in_end_effector_frame.params["root_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.fixed_asset_in_end_effector_frame.params["target_asset_offset"] = KEYPOINTS_BOLTM16.bolt_tip_offset

        # critic
        self.critic.end_effector_pose.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.critic.end_effector_vel_lin_ang_b.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.critic.held_asset_pose.params["target_asset_cfg"] = SceneEntityCfg("nut_m16")
        self.critic.fixed_asset_pose.params["target_asset_cfg"] = SceneEntityCfg("bolt_m16")
        self.critic.held_asset_in_fixed_asset_frame.params["target_asset_cfg"] = SceneEntityCfg("nut_m16")
        self.critic.held_asset_in_fixed_asset_frame.params["root_asset_cfg"] = SceneEntityCfg("bolt_m16")
        self.critic.held_asset_in_fixed_asset_frame.params["root_asset_offset"] = KEYPOINTS_BOLTM16.bolt_tip_offset


@configclass
class NutThreadEventCfg(FactoryEventCfg):
    def __post_init__(self):
        # For asset_material
        self.held_asset_material.params["asset_cfg"] = SceneEntityCfg("nut_m16")
        self.fixed_asset_material.params["asset_cfg"] = SceneEntityCfg("bolt_m16")

        # For reset_fixed_asset
        self.reset_fixed_asset.params["asset_list"] = ["bolt_m16"]

        # For reset_hand
        self.reset_end_effector_around_fixed_asset.params["fixed_asset_cfg"] = SceneEntityCfg("bolt_m16")
        self.reset_end_effector_around_fixed_asset.params["fixed_asset_offset"] = KEYPOINTS_BOLTM16.bolt_tip_offset
        self.reset_end_effector_around_fixed_asset.params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
        self.reset_end_effector_around_fixed_asset.params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
        self.reset_end_effector_around_fixed_asset.params["pose_range_b"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.015, 0.025),
            "roll": (3.141, 3.141),
            "yaw": (1.57, 2.09),
        }

        # For reset_held_asset
        self.reset_held_asset_in_hand.params["holding_body_cfg"].body_names = "panda_fingertip_centered"
        self.reset_held_asset_in_hand.params["held_asset_cfg"] = SceneEntityCfg("nut_m16")
        self.reset_held_asset_in_hand.params["held_asset_graspable_offset"] = KEYPOINTS_NUTM16.grasp_point

        # For grasp_held_assset
        self.grasp_held_asset.params["robot_cfg"].body_names = "panda_fingertip_centered"
        self.grasp_held_asset.params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
        self.grasp_held_asset.params["held_asset_diameter"] = KEYPOINTS_NUTM16.grasp_diameter


@configclass
class NutThreadRewardsCfg(FactoryRewardsCfg):
    def __post_init__(self):
        # For progress_context
        self.progress_context.params["fixed_asset_cfg"] = SceneEntityCfg("bolt_m16")
        self.progress_context.params["held_asset_cfg"] = SceneEntityCfg("nut_m16")
        self.progress_context.params["held_asset_offset"] = KEYPOINTS_NUTM16.center_axis_bottom
        self.progress_context.params["fixed_asset_offset"] = KEYPOINTS_BOLTM16.screwed_nut_offset


@configclass
class NutThreadEnvCfg(FactoryBaseEnvCfg):
    """Configuration for the NutThread environment."""

    observations: NutThreadObservationsCfg = NutThreadObservationsCfg()
    events: NutThreadEventCfg = NutThreadEventCfg()
    rewards: NutThreadRewardsCfg = NutThreadRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        for asset in ["gear_base", "hole_8mm", "small_gear", "large_gear", "medium_gear", "peg_8mm"]:
            delattr(self.scene, asset)
