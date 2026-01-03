# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp

OBJECT_SPAWN_HEIGHT = 0.5


@configclass
class PartialAssembliesSceneCfg(InteractiveSceneCfg):
    """Scene configuration for partial assemblies environment."""

    insertive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=True,
                kinematic_enabled=False,
            ),
            # assume very light
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT * 2), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    receptive_object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd",
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            # since kinematic_enabled=True, mass does not matter
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
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
class PartialAssembliesEventCfg:
    """Configuration for partial assemblies randomization."""

    # Low friction so that the object can around
    insertive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.0, 0.0),
            "dynamic_friction_range": (0.0, 0.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "make_consistent": True,
        },
    )

    receptive_object_material = EventTerm(
        func=task_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "static_friction_range": (0.0, 0.0),
            "dynamic_friction_range": (0.0, 0.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "make_consistent": True,
        },
    )

    partial_assembly_sampling = EventTerm(
        func=task_mdp.assembly_sampling_event,
        mode="reset",
        params={
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
        },
    )

    apply_forces = EventTerm(
        func=task_mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(1 / 120, 1 / 120),
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "force_range": (-0.005, 0.005),
            "torque_range": (-0.005, 0.005),
        },
    )

    # Collect pose data from environments with positive rewards
    pose_data_collection = EventTerm(
        func=task_mdp.pose_logging_event,
        mode="interval",
        interval_range_s=(1 / 120, 1 / 120),
        params={
            "receptive_object_cfg": SceneEntityCfg("receptive_object"),
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
        },
    )


@configclass
class PartialAssembliesTerminationCfg:
    """Configuration for partial assemblies termination conditions."""

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    obb_no_overlap = DoneTerm(
        func=task_mdp.check_obb_no_overlap_termination,
        params={
            "insertive_object_cfg": SceneEntityCfg("insertive_object"),
            "enable_visualization": False,
        },
        time_out=True,
    )


@configclass
class PartialAssembliesObservationsCfg:
    """Configuration for partial assemblies observations."""

    pass


@configclass
class PartialAssembliesRewardsCfg:
    """Configuration for partial assemblies rewards."""

    collision_free = RewTerm(
        func=task_mdp.collision_free,
        params={
            "collision_analyzer_cfg": task_mdp.CollisionAnalyzerCfg(
                num_points=1024,
                max_dist=0.5,
                min_dist=-0.0005,
                asset_cfg=SceneEntityCfg("insertive_object"),
                obstacle_cfgs=[SceneEntityCfg("receptive_object")],
            )
        },
        weight=1.0,
    )


@configclass
class PartialAssembliesActionsCfg:
    """Configuration for partial assemblies actions."""

    pass


def make_insertive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.001),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT * 2), rot=(1.0, 0.0, 0.0, 0.0)),
    )


def make_receptive_object(usd_path: str):
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ReceptiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
                disable_gravity=False,
                kinematic_enabled=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, OBJECT_SPAWN_HEIGHT), rot=(1.0, 0.0, 0.0, 0.0)),
    )


variants = {
    "scene.insertive_object": {
        "fbleg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
        "fbdrawerbottom": make_insertive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
        ),
        "peg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
        "cupcake": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
        "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
        "rectangle": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
    },
    "scene.receptive_object": {
        "fbtabletop": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareTableTop/square_table_top.usd"
        ),
        "fbdrawerbox": make_receptive_object(
            f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBox/drawer_box.usd"
        ),
        "peghole": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/PegHole/peg_hole.usd"),
        "plate": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Plate/plate.usd"),
        "cube": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/ReceptiveCube/receptive_cube.usd"),
        "wall": make_receptive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Wall/wall.usd"),
    },
}


@configclass
class PartialAssembliesCfg(ManagerBasedRLEnvCfg):
    """Configuration for partial assemblies environment without robot."""

    scene: PartialAssembliesSceneCfg = PartialAssembliesSceneCfg(num_envs=1, env_spacing=2.0)
    events: PartialAssembliesEventCfg = PartialAssembliesEventCfg()
    terminations: PartialAssembliesTerminationCfg = PartialAssembliesTerminationCfg()
    observations: PartialAssembliesObservationsCfg = PartialAssembliesObservationsCfg()
    actions: PartialAssembliesActionsCfg = PartialAssembliesActionsCfg()
    rewards: PartialAssembliesRewardsCfg = PartialAssembliesRewardsCfg()
    viewer: ViewerCfg = ViewerCfg(eye=(2.0, 0.0, 0.75), origin_type="world", env_index=0, asset_name="receptive_object")
    variants = variants

    def __post_init__(self):
        self.decimation = 1  # We want to save fine-grained poses
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
