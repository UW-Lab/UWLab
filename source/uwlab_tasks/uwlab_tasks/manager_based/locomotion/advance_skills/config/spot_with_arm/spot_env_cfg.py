# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.spot as spot

import uwlab_tasks.manager_based.locomotion.advance_skills.config.spot.mdp as spot_mdp

from ... import advance_skills_base_env, advance_skills_env
from ... import mdp as mdp


@configclass
class SpotActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["^(?!.*arm0).*$"], scale=0.2, use_default_offset=True
    )
    arm_pos = mdp.DefaultJointPositionStaticActionCfg(
        asset_name="robot", joint_names=["arm0.*"], scale=1, use_default_offset=True
    )


@configclass
class SportRewardsCfg(advance_skills_base_env.RewardsCfg):
    move_forward = RewTerm(
        func=spot_mdp.reward_forward_velocity,
        weight=0.3,
        params={
            "std": 1,
            "max_iter": 200,
        },
    )

    air_time = RewTerm(
        func=spot_mdp.air_time_reward,
        weight=1.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )

    gait = RewTerm(
        func=spot_mdp.GaitReward,
        weight=2.0,
        params={
            "std": 0.1,
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "synced_feet_pair_names": (("fl_foot", "hr_foot"), ("fr_foot", "hl_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
            "max_iterations": 400,
        },
    )

    # -- penalties
    air_time_variance = RewTerm(
        func=spot_mdp.air_time_variance_penalty,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    foot_slip = RewTerm(
        func=spot_mdp.foot_slip_penalty,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 1.0,
        },
    )


@configclass
class SpotEnvMixin:
    actions: SpotActionsCfg = SpotActionsCfg()
    rewards: SportRewardsCfg = SportRewardsCfg()

    def __post_init__(self: advance_skills_base_env.AdvanceSkillsBaseEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()
        # overwrite as spot's body names for sensors
        self.scene.robot = spot.SPOT_WITH_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body"
        self.scene.height_scanner.pattern_cfg.resolution = 0.15
        self.scene.height_scanner.pattern_cfg.size = (3.5, 1.5)

        # overwrite as spot's body names for events
        self.events.add_base_mass.params["asset_cfg"].body_names = "body"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "body"

        self.rewards.undesired_contact.params["sensor_cfg"].body_names = ["body", ".*leg"]
        self.rewards.feet_lin_acc_l2.params["robot_cfg"].body_names = ".*_foot"
        self.rewards.feet_rot_acc_l2.params["robot_cfg"].body_names = ".*_foot"
        self.rewards.illegal_contact_penalty.params["sensor_cfg"].body_names = "body"

        self.terminations.base_contact.params["sensor_cfg"].body_names = "body"
        self.viewer.body_name = "body"

        self.sim.dt = 0.002
        self.decimation = 10


@configclass
class AdvanceSkillsSpotEnvCfg(SpotEnvMixin, advance_skills_env.AdvanceSkillsEnvCfg):
    pass


@configclass
class PitSpotEnvCfg(SpotEnvMixin, advance_skills_env.PitEnvConfig):
    pass


@configclass
class GapSpotEnvCfg(SpotEnvMixin, advance_skills_env.GapEnvConfig):
    pass


@configclass
class SlopeInvSpotEnvCfg(SpotEnvMixin, advance_skills_env.SlopeInvEnvConfig):
    pass


@configclass
class ExtremeStairSpotEnvCfg(SpotEnvMixin, advance_skills_env.ExtremeStairEnvConfig):
    pass


@configclass
class SquarePillarObstacleSpotEnvCfg(SpotEnvMixin, advance_skills_env.SquarePillarObstacleEnvConfig):
    pass


@configclass
class IrregularPillarObstacleSpotEnvCfg(SpotEnvMixin, advance_skills_env.IrregularPillarObstacleEnvConfig):
    pass
