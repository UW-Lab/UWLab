# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.spot as spot

import uwlab_tasks.manager_based.locomotion.risky_terrains.mdp as mdp

from ... import balance_beams_env, stepping_beams_env, stepping_stones_env


@configclass
class SpotActionsCfg:
    actions = spot.SPOT_JOINT_POSITION


@configclass
class SpotRewardsCfg(stepping_stones_env.RewardsCfg):
    joint_torque_limits = RewTerm(
        func=mdp.torque_limits,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_h[xy]"),
            "actuator_name": "spot_hip",
            "ratio": 1.0,
        },
    )

    joint_torque_limits_knee = RewTerm(
        func=mdp.torque_limits_knee,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_kn"),
            "ratio": 1.0,
        },
    )

    move_forward = RewTerm(
        func=mdp.reward_forward_velocity,
        weight=0.3,
        params={
            "std": 1,
            "max_iter": 300,
            "init_amplification": 10,
            "forward_vector": [1.0, 0.0, 0.0],
        },
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

    foot_on_ground = RewTerm(
        func=mdp.foot_on_ground,
        weight=5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "d": 0.5, "tr": 1.0},
    )

    gait = RewTerm(
        func=mdp.GaitReward,
        weight=8.0,
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

    joint_pos = RewTerm(
        func=mdp.joint_position_penalty,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 2.5,
            "velocity_threshold": 0.5,
        },
    )

    def __post_init__(self):
        self.undesired_contact.params["sensor_cfg"].body_names = [".*_uleg", ".*_lleg"]
        self.contact_force_pen.params["sensor_cfg"].body_names = ".*_foot"
        self.feet_accel.params["robot_cfg"].body_names = ".*_foot"
        self.move_in_dir.params["max_iter"] = 200

        self.base_accel.params["robot_cfg"].body_names = "body"


@configclass
class SpotEnvMixin:
    actions: SpotActionsCfg = SpotActionsCfg()
    rewards: SpotRewardsCfg = SpotRewardsCfg()

    def __post_init__(self: stepping_stones_env.SteppingStoneLocomotionEnvCfg):
        # Ensure parent classes run their setup first
        super().__post_init__()  # type: ignore

        self.decimation = 10
        self.sim.dt = 0.002

        # overwrite as spot's body names for sensors
        self.scene.robot = spot.SPOT_WITH_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/body"
        self.scene.height_scanner.pattern_cfg.resolution = 0.15
        self.scene.height_scanner.pattern_cfg.size = (3.5, 1.5)

        # overwrite as spot's body names for events
        self.events.add_base_mass.params["asset_cfg"].body_names = "body"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "body"

        self.terminations.base_contact.params["sensor_cfg"].body_names = "body"
        self.viewer.body_name = "body"


@configclass
class SteppingStoneSpotEnvCfg(SpotEnvMixin, stepping_stones_env.SteppingStoneLocomotionEnvCfg):
    pass


@configclass
class BalanceBeamsSpotEnvCfg(SpotEnvMixin, balance_beams_env.BalanceBeamsLocomotionEnvCfg):
    pass


@configclass
class SteppingBeamsSpotEnvCfg(SpotEnvMixin, stepping_beams_env.SteppingBeamsLocomotionEnvCfg):
    pass
