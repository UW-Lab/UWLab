# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import uwlab_tasks.manager_based.locomotion.risky_terrains.mdp as mdp

from . import stepping_stones_env
from .config.terrains.terrain_cfg import STEPPING_BEAMS_CFG


@configclass
class SteppingBeamSceneCfg(stepping_stones_env.SteppingStoneSceneCfg):
    def __post_init__(self):
        self.terrain.terrain_generator = STEPPING_BEAMS_CFG


@configclass
class CommandsCfg:
    """Commands for the MDP."""

    target_cmd = mdp.UniformPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10, 10),
        debug_vis=True,
        simple_heading=False,
        ranges=mdp.UniformPose2dCommandCfg.Ranges(pos_x=(2.5, 4.5), pos_y=(-0.1, 0.1), heading=(-0.785, 0.785)),
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

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


@configclass
class RewardsCfg(stepping_stones_env.RewardsCfg):
    """Rewards for the MDP."""

    # Balance Beam specific rewards

    aggressive_motion = RewTerm(func=mdp.aggressive_motion, weight=-5.0, params={"threshold": 1.0})

    stand_pose = RewTerm(func=mdp.stand_pos, weight=-5.0, params={"base_height": 0.6, "tr": 1.0, "phi": 0.5, "d": 0.25})

    def __post_init__(self):
        self.position_tracking.weight = 25.0
        self.head_tracking.weight = 12.0
        self.joint_torque_limits.weight = -0.5
        self.stand_still.params["d"] = 0.25


@configclass
class CurriculumCfg:
    """Curriculum for the MDP."""

    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_risky,  # type: ignore
        params={
            "demotion_fraction": 0.00,
        },
    )


@configclass
class SteppingBeamsLocomotionEnvCfg(stepping_stones_env.SteppingStoneLocomotionEnvCfg):
    scene: SteppingBeamSceneCfg = SteppingBeamSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
