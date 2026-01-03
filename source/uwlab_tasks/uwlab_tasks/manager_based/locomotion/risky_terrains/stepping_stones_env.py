# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import uwlab_tasks.manager_based.locomotion.risky_terrains.mdp as mdp

from .config.terrains.terrain_cfg import RISKY_TERRAINS_CFG


@configclass
class SteppingStoneSceneCfg(InteractiveSceneCfg):

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=RISKY_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # robots
    robot: ArticulationCfg = MISSING  # type: ignore

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.07, size=(1.68, 1.05)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, debug_vis=True
    )


@configclass
class ActionsCfg:
    """Actions for the MDP."""

    pass


@configclass
class CommandsCfg:
    """Commands for the MDP."""

    target_cmd = mdp.UniformPolarPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(10, 10),
        debug_vis=True,
        simple_heading=False,
        ranges=mdp.UniformPolarPose2dCommandCfg.Ranges(distance_range=(1.5, 4.9), heading=(-3.14, 3.14)),
    )


@configclass
class ObservationsCfg:
    """Observations for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        concatenate_cmd = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_cmd"})
        time_left = ObsTerm(func=mdp.time_left)
        joint_pos = ObsTerm(func=mdp.joint_pos, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_actions = ObsTerm(func=mdp.last_action)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg(name="height_scanner")},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    """Events for the MDP."""

    # startup
    physical_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
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

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # interval
    # comment for pit and gap
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(3.0, 4.5),
    #     params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    # )

    reset_episode_length = EventTerm(
        func=mdp.reset_episode_length_s, mode="reset", params={"episode_length_s": (5.0, 7.0)}
    )


@configclass
class RewardsCfg:
    """Rewards for the MDP."""

    position_tracking = RewTerm(func=mdp.position_tracking, weight=10.0, params={"tr": 2.0})
    head_tracking = RewTerm(func=mdp.heading_tracking, weight=5, params={"d": 2.0, "tr": 4.0})
    early_termination = RewTerm(
        func=mdp.is_terminated,
        weight=-200,
    )

    # penalties
    undesired_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*THIGH", ".*SHANK"]),
            "threshold": 1.0,
        },
    )

    # illegal_contact_penalty = RewTerm(
    #     func=mdp.illegal_contact_penalty,
    #     weight=-1,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    # )

    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)

    joint_vel_limit = RewTerm(func=mdp.joint_vel_limit_pen, weight=-1.0, params={"limits_factor": 0.9})

    base_accel = RewTerm(
        func=mdp.base_accel_pen,
        weight=-0.001,
        params={"ratio": 0.02, "robot_cfg": SceneEntityCfg("robot", body_names="base")},
    )

    feet_accel = RewTerm(
        func=mdp.feet_accel_l1_pen, weight=-0.0005, params={"robot_cfg": SceneEntityCfg("robot", body_names=".*FOOT")}
    )

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    joint_torque_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)

    joint_torque_limits = RewTerm(func=mdp.torque_limits, weight=-0.2, params={"ratio": 1.0})

    contact_force_pen = RewTerm(
        func=mdp.contact_forces_pen,
        weight=-2.5e-5,
        params={"threshold": 700, "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*FOOT")},
    )

    dont_wait = RewTerm(
        func=mdp.dont_wait,
        weight=-1.0,
        params={"robot_cfg": SceneEntityCfg("robot"), "d": 1.0, "velocity_threshold": 0.2},
    )

    move_in_dir = RewTerm(
        func=mdp.move_in_dir,
        weight=1.0,
        params={
            "max_iter": 150,
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-1.0,
        params={"robot_cfg": SceneEntityCfg("robot"), "d": 0.5, "tr": 1.0, "phi": 0.5},
    )

    # curiosity_reward = RewTerm(
    #     func=mdp.CuriosityReward,
    #     weight=1.0,
    #     params={
    #         "optimization_weight": 1.0,
    #         "lr": 1e-3
    #     }
    # )


@configclass
class TerminationsCfg:
    """Terminations for the MDP."""

    time_out = DoneTerm(func=mdp.custom_time_out, time_out=True)

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )

    ill_posture = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.75})


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
class SteppingStoneLocomotionEnvCfg(ManagerBasedRLEnvCfg):
    scene: SteppingStoneSceneCfg = SteppingStoneSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(1.0, 2.0, 2.0), origin_type="asset_body", asset_name="robot", body_name="base")

    def __post_init__(self):
        self.is_finite_horizon = True
        self.decimation = 2
        self.episode_length_s = 6.0

        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**24
        self.sim.physx.gpu_collision_stack_size = 2**27
        self.sim.physx.gpu_max_rigid_patch_count = 6 * 2**15

        self.viewer.resolution = (1920, 1080)

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False
