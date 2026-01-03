# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

import uwlab_assets.robots.ur5e_robotiq_gripper as ur5e_robotiq_gripper
from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp

OBJECT_SPAWN_HEIGHT = 0.5


@configclass
class GraspSamplingSceneCfg(InteractiveSceneCfg):
    """Scene configuration for grasp sampling environment."""

    robot = ur5e_robotiq_gripper.ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0, disable_gravity=False
            ),
            # assume very light
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # Environment
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class GraspSamplingEventCfg:
    """Configuration for grasp sampling randomization."""

    reset_object_position = EventTerm(
        func=task_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.3, 0.3),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )

    grasp_sampling = EventTerm(
        func=task_mdp.grasp_sampling_event,
        mode="reset",
        params={
            "object_cfg": SceneEntityCfg("object"),
            "gripper_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            "num_candidates": 1e6,
            "num_standoff_samples": 32,
            "num_orientations": 16,
            "lateral_sigma": 0.0,
            "visualize_grasps": False,
            "visualization_scale": 0.01,
        },
    )

    global_physics_control_event = EventTerm(
        func=task_mdp.global_physics_control_event,
        mode="interval",
        interval_range_s=(0.1, 0.1),
        params={
            "gravity_on_interval": (1.0, np.inf),
            "force_torque_on_interval": (1.0, 2.0),
            "force_torque_asset_cfgs": [SceneEntityCfg("object")],
            "force_torque_magnitude": 0.01,
        },
    )


@configclass
class GraspSamplingTerminationCfg:
    """Configuration for grasp sampling termination conditions."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    success = DoneTerm(
        func=task_mdp.check_grasp_success,
        params={
            "object_cfg": SceneEntityCfg("object"),
            "gripper_cfg": SceneEntityCfg("robot"),
            "collision_analyzer_cfg": task_mdp.CollisionAnalyzerCfg(
                num_points=1024,
                max_dist=0.5,
                min_dist=-0.0005,
                asset_cfg=SceneEntityCfg("robot"),
                obstacle_cfgs=[SceneEntityCfg("object")],
            ),
            "max_pos_deviation": OBJECT_SPAWN_HEIGHT / 2,
            "pos_z_threshold": OBJECT_SPAWN_HEIGHT / 2,
        },
        time_out=True,
    )


@configclass
class GraspSamplingObservationsCfg:
    """Configuration for grasp sampling observations."""

    pass


@configclass
class GraspSamplingRewardsCfg:
    """Configuration for grasp sampling rewards."""

    pass


def make_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.object": {
        "fbleg": make_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
        "fbdrawerbottom": make_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
        ),
        "peg": make_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "cupcake": make_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cube": make_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
        "rectangle": make_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
    }
}


@configclass
class Robotiq2f85GraspSamplingCfg(ManagerBasedRLEnvCfg):
    """Configuration for grasp sampling environment with Robotiq 2F85 gripper."""

    scene: GraspSamplingSceneCfg = GraspSamplingSceneCfg(num_envs=1, env_spacing=1.5)
    events: GraspSamplingEventCfg = GraspSamplingEventCfg()
    terminations: GraspSamplingTerminationCfg = GraspSamplingTerminationCfg()
    observations: GraspSamplingObservationsCfg = GraspSamplingObservationsCfg()
    actions: ur5e_robotiq_gripper.Robotiq2f85BinaryGripperAction = ur5e_robotiq_gripper.Robotiq2f85BinaryGripperAction()
    rewards: GraspSamplingRewardsCfg = GraspSamplingRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="robot")
    variants = variants

    def __post_init__(self):
        self.decimation = 12
        self.episode_length_s = 4.0
        # simulation settings
        self.sim.dt = 1 / 120.0

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 192
        self.sim.physx.max_velocity_iteration_count = 1
        self.sim.physx.bounce_threshold_velocity = 0.02
        self.sim.physx.friction_offset_threshold = 0.01
        self.sim.physx.friction_correlation_distance = 0.0005

        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_collision_stack_size = 2**31

        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
