# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import uwlab_tasks.manager_based.manipulation.factory_extension.mdp as mdp

from . import factory_assets_cfg as assets

"""
Base scene definition for Factory Tasks
"""


@configclass
class FactorySceneCfg(InteractiveSceneCfg):
    """Configuration for a factory task scene."""

    # Ground plane
    ground = assets.GROUND_CFG

    # Table
    table = assets.TABLE_CFG

    # NIST Board
    nistboard = assets.NISTBOARD_CFG

    # "FIXED ASSETS"
    bolt_m16: ArticulationCfg = assets.BOLT_M16_CFG
    gear_base: ArticulationCfg = assets.GEAR_BASE_CFG
    hole_8mm: ArticulationCfg = assets.HOLE_8MM_CFG

    # "Moving Gears"
    small_gear: ArticulationCfg = assets.SMALL_GEAR_CFG
    large_gear: ArticulationCfg = assets.LARGE_GEAR_CFG

    # "HELD ASSETS"
    nut_m16: ArticulationCfg = assets.NUT_M16_CFG
    medium_gear: ArticulationCfg = assets.MEDIUM_GEAR_CFG
    peg_8mm: ArticulationCfg = assets.PEG_8MM_CFG

    # Robot Override
    robot: ArticulationCfg = MISSING  # type: ignore

    # Lights
    dome_light = assets.DOMELIGHT_CFG


@configclass
class FactoryObservationsCfg:
    """Observation specifications for Factory."""

    @configclass
    class PolicyCfg(ObsGroup):
        end_effector_vel_lin_ang_b = ObsTerm(
            func=mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        end_effector_pose = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        fixed_asset_in_end_effector_frame: ObsTerm = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("fixed_asset"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            },
        )

        prev_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        prev_actions = ObsTerm(func=mdp.last_action)

        joint_pos = ObsTerm(func=mdp.joint_pos)

        end_effector_pose = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        end_effector_vel_lin_ang_b = ObsTerm(
            func=mdp.asset_link_velocity_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "root_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        held_asset_pose = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={"target_asset_cfg": SceneEntityCfg("held_asset"), "root_asset_cfg": SceneEntityCfg("robot")},
        )

        fixed_asset_pose = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={"target_asset_cfg": SceneEntityCfg("fixed_asset"), "root_asset_cfg": SceneEntityCfg("robot")},
        )

        held_asset_in_fixed_asset_frame: ObsTerm = ObsTerm(
            func=mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("held_asset"),
                "root_asset_cfg": SceneEntityCfg("fixed_asset"),
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class FactoryEventCfg:
    """Events specifications for Factory"""

    # mode: startup
    held_asset_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("held_asset"),
        },
    )

    fixed_asset_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("fixed_asset"),
        },
    )

    robot_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # mode: reset
    reset_env = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_board = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("nistboard"),
        },
    )

    reset_fixed_asset = EventTerm(
        func=mdp.reset_fixed_assets,
        mode="reset",
        params={
            "asset_list": ["fixed_asset"],
        },
    )

    reset_end_effector_around_fixed_asset = EventTerm(
        func=mdp.reset_end_effector_round_fixed_asset,  # type: ignore
        mode="reset",
        params={
            "fixed_asset_cfg": MISSING,
            "fixed_asset_offset": MISSING,
            "pose_range_b": MISSING,
            "robot_ik_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_held_asset_in_hand = EventTerm(
        func=mdp.reset_held_asset,
        mode="reset",
        params={
            "holding_body_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "held_asset_graspable_offset": MISSING,
            "held_asset_inhand_range": {},
        },
    )

    grasp_held_asset = EventTerm(
        func=mdp.grasp_held_asset,
        mode="reset",
        params={"robot_cfg": SceneEntityCfg("robot", body_names="end_effector"), "held_asset_diameter": MISSING},
    )


@configclass
class FactoryRewardsCfg:
    """Reward terms for Factory"""

    # penalties
    action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0)

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0)

    # progress rewards
    progress_context = RewTerm(
        func=mdp.ProgressContext,  # type: ignore
        weight=0.01,
        params={
            "success_threshold": 0.001,
            "held_asset_cfg": SceneEntityCfg("held_asset"),
            "fixed_asset_cfg": SceneEntityCfg("fixed_asset"),
            "held_asset_offset": MISSING,
            "fixed_asset_offset": MISSING,
        },
    )

    # orientation_alignment = RewTerm(func=mdp.orientation_reward, weight=1.0, params={"std": 0.02})

    concentric_alignment_coarse = RewTerm(func=mdp.concentric_reward, weight=5.0, params={"std": 0.02})

    concentric_alignment_fine = RewTerm(func=mdp.concentric_reward, weight=10.0, params={"std": 0.005})

    progress_reward_coarse = RewTerm(func=mdp.progress_reward, weight=5.0, params={"std": 0.02})

    progress_reward_fine = RewTerm(func=mdp.progress_reward, weight=10.0, params={"std": 0.005})

    success_reward = RewTerm(func=mdp.success_reward, weight=20.0)


@configclass
class FactoryTerminationsCfg:
    """Termination terms for Factory."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


##
# Environment configuration
##
@configclass
class FactoryBaseEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the base Factory environment."""

    scene: FactorySceneCfg = FactorySceneCfg(num_envs=2, env_spacing=2.0)
    observations: FactoryObservationsCfg = FactoryObservationsCfg()
    events: FactoryEventCfg = FactoryEventCfg()
    terminations: FactoryTerminationsCfg = FactoryTerminationsCfg()
    rewards: FactoryRewardsCfg = FactoryRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.5, 0.1), origin_type="asset_body", asset_name="robot", body_name="panda_fingertip_centered"
    )
    actions = MISSING

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 8  # 15hz
        self.episode_length_s = 10
        # simulation settings
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation

        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192  # Important to avoid interpenetration.
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31
        self.sim.physx.gpu_max_num_partitions = 1

        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0

        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_dlssg = True
