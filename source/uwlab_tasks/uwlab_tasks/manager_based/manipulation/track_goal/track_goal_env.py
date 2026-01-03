# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import time_out
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from . import mdp


class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING  # type: ignore

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.0, -0.868), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.UsdFileCfg(usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd"),
    )

    # override ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
        spawn=sim_utils.GroundPlaneCfg(),
    )
    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.75, 0.75, 0.75))
    )


@configclass
class ObservationsCfg:
    """"""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    pass


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="REPLACE_ME",
        resampling_time_range=(1.5, 2.5),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.6),
            pos_y=(-0.325, 0.325),
            pos_z=(0.125, 0.5),
            roll=(0, 0.0),
            pitch=(0, 1),
            yaw=(0.0, 1),
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    end_effector_position_tracking = RewTerm(
        func=mdp.link_position_command_align_tanh,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="REPLACE_ME"),
            "std": 0.5,
            "command_name": "ee_pose",
        },
    )

    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.link_position_command_align_tanh,
        weight=1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="REPLACE_ME"),
            "std": 0.1,
            "command_name": "ee_pose",
        },
    )

    end_effector_orientation_tracking = RewTerm(
        func=mdp.link_orientation_command_align_tanh,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="REPLACE_ME"),
            "std": 0.5,
            "command_name": "ee_pose",
        },
    )

    end_effector_orientation_tracking_fine_grained = RewTerm(
        func=mdp.link_orientation_command_align_tanh,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="REPLACE_ME"),
            "std": 0.05,
            "command_name": "ee_pose",
        },
    )


@configclass
class EventCfg:
    reset_everything = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_robot_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": [-0.2, 0.2],
            "velocity_range": [-0.1, 0.1],
        },
        mode="reset",
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


@configclass
class TrackGoalEnv(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = MISSING  # type: ignore
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 50
        # simulation settings
        self.sim.dt = 0.02 / self.decimation
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**16

        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dlssg = True
