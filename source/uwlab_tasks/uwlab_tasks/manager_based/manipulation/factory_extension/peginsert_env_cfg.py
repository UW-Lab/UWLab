# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .assembly_keypoints import KEYPOINTS_HOLE8MM, KEYPOINTS_PEG8MM
from .factory_env_base import FactoryBaseEnvCfg, FactoryEventCfg, FactoryObservationsCfg, FactoryRewardsCfg


@configclass
class PegInsertObservationsCfg(FactoryObservationsCfg):
    def __post_init__(self):
        # policy
        self.policy.end_effector_vel_lin_ang_b.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.end_effector_pose.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.fixed_asset_in_end_effector_frame.params["target_asset_cfg"] = SceneEntityCfg("hole_8mm")
        self.policy.fixed_asset_in_end_effector_frame.params["root_asset_cfg"].body_names = "panda_fingertip_centered"
        self.policy.fixed_asset_in_end_effector_frame.params["target_asset_offset"] = KEYPOINTS_HOLE8MM.hole_tip_offset

        # critic
        self.critic.end_effector_pose.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.critic.end_effector_vel_lin_ang_b.params["target_asset_cfg"].body_names = "panda_fingertip_centered"
        self.critic.held_asset_pose.params["target_asset_cfg"] = SceneEntityCfg("peg_8mm")
        self.critic.fixed_asset_pose.params["target_asset_cfg"] = SceneEntityCfg("hole_8mm")
        self.critic.held_asset_in_fixed_asset_frame.params["target_asset_cfg"] = SceneEntityCfg("peg_8mm")
        self.critic.held_asset_in_fixed_asset_frame.params["root_asset_cfg"] = SceneEntityCfg("hole_8mm")
        self.critic.held_asset_in_fixed_asset_frame.params["root_asset_offset"] = KEYPOINTS_HOLE8MM.hole_tip_offset


@configclass
class PegInsertEventCfg(FactoryEventCfg):
    def __post_init__(self):
        # For asset_material
        self.held_asset_material.params["asset_cfg"] = SceneEntityCfg("peg_8mm")
        self.fixed_asset_material.params["asset_cfg"] = SceneEntityCfg("hole_8mm")

        # For reset_fixed_asset
        self.reset_fixed_asset.params["asset_list"] = ["hole_8mm"]

        # For reset_end_effector_around_fixed_asset
        self.reset_end_effector_around_fixed_asset.params["fixed_asset_cfg"] = SceneEntityCfg("hole_8mm")
        self.reset_end_effector_around_fixed_asset.params["fixed_asset_offset"] = KEYPOINTS_HOLE8MM.hole_tip_offset
        self.reset_end_effector_around_fixed_asset.params["robot_ik_cfg"].joint_names = ["panda_joint.*"]
        self.reset_end_effector_around_fixed_asset.params["robot_ik_cfg"].body_names = "panda_fingertip_centered"
        self.reset_end_effector_around_fixed_asset.params["pose_range_b"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "z": (0.047, 0.057),
            "roll": (3.141, 3.141),
            "yaw": (-0.785, 0.785),
        }

        # For reset_held_asset
        self.reset_held_asset_in_hand.params["holding_body_cfg"].body_names = "panda_fingertip_centered"
        self.reset_held_asset_in_hand.params["held_asset_cfg"] = SceneEntityCfg("peg_8mm")
        self.reset_held_asset_in_hand.params["held_asset_graspable_offset"] = KEYPOINTS_PEG8MM.grasp_point

        # For grasp_held_assset
        self.grasp_held_asset.params["robot_cfg"].body_names = "panda_fingertip_centered"
        self.grasp_held_asset.params["robot_cfg"].joint_names = "panda_finger_joint[1-2]"
        self.grasp_held_asset.params["held_asset_diameter"] = KEYPOINTS_PEG8MM.grasp_diameter


@configclass
class PegInsertRewardsCfg(FactoryRewardsCfg):
    def __post_init__(self):
        # For progress_context
        self.progress_context.params["fixed_asset_cfg"] = SceneEntityCfg("hole_8mm")
        self.progress_context.params["held_asset_cfg"] = SceneEntityCfg("peg_8mm")
        self.progress_context.params["held_asset_offset"] = KEYPOINTS_PEG8MM.center_axis_bottom
        self.progress_context.params["fixed_asset_offset"] = KEYPOINTS_HOLE8MM.inserted_peg_base_offset


@configclass
class PegInsertEnvCfg(FactoryBaseEnvCfg):
    """Configuration for the PegInsert environment."""

    observations: PegInsertObservationsCfg = PegInsertObservationsCfg()
    events: PegInsertEventCfg = PegInsertEventCfg()
    rewards: PegInsertRewardsCfg = PegInsertRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        for asset in ["bolt_m16", "gear_base", "small_gear", "large_gear", "medium_gear", "nut_m16"]:
            delattr(self.scene, asset)
