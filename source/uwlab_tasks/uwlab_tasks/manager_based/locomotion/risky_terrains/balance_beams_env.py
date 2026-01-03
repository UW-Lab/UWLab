# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import uwlab_tasks.manager_based.locomotion.risky_terrains.mdp as mdp

from . import stepping_stones_env
from .config.terrains.terrain_cfg import BALANCE_BEAMS_CFG


@configclass
class BalanceBeamSceneCfg(stepping_stones_env.SteppingStoneSceneCfg):
    def __post_init__(self):
        self.terrain.terrain_generator = BALANCE_BEAMS_CFG


@configclass
class CommandsCfg:
    """Commands for the MDP."""

    # for balance-beams
    target_cmd = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10, 10),
        debug_vis=True,
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(3.5, 4.5), pos_y=(-0.1, 0.1), heading=(-0.785, 0.785)),
    )


@configclass
class ObservationsCfg:
    """Observations for the MDP."""

    @configclass
    class PolicyCfg(stepping_stones_env.ObservationsCfg.PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.height_scan.noise = Unoise(n_min=-0.025, n_max=0.025)

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg(stepping_stones_env.EventsCfg):
    """Events for the MDP."""

    def __post_init__(self):
        self.reset_episode_length.params["episode_length_s"] = (8.0, 10.0)
        self.reset_base.params["pose_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}


@configclass
class RewardsCfg(stepping_stones_env.RewardsCfg):
    """Rewards for the MDP."""

    # balance beam specific
    aggressive_motion = RewTerm(func=mdp.aggressive_motion, weight=-5.0, params={"threshold": 1.0})

    stand_pose = RewTerm(func=mdp.stand_pos, weight=-5.0, params={"base_height": 0.6, "tr": 1.0, "phi": 0.5, "d": 0.25})

    def __post_init__(self):
        self.position_tracking.weight = 25.0
        self.head_tracking.weight = 12.0
        self.joint_torque_limits.weight = -0.5
        self.joint_torque_limits.params["ratio"] = 0.8
        self.stand_still.params["d"] = 0.25


@configclass
class BalanceBeamsLocomotionEnvCfg(stepping_stones_env.SteppingStoneLocomotionEnvCfg):
    scene: BalanceBeamSceneCfg = BalanceBeamSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
