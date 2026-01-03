# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

# This is where we will get the Robot that we want to use
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

ASSET_DIR = f"{ISAACLAB_NUCLEUS_DIR}/Factory"


GROUND_CFG = AssetBaseCfg(
    prim_path="/World/ground",
    spawn=sim_utils.GroundPlaneCfg(),
    init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.868)),
)

DOMELIGHT_CFG = AssetBaseCfg(
    prim_path="/World/DomeLight",
    spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=1500.0),
)


FRANKA_PANDA_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/franka_mimic.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "panda_joint1": 0.00871,
            "panda_joint2": -0.10368,
            "panda_joint3": -0.00794,
            "panda_joint4": -1.49139,
            "panda_joint5": -0.00083,
            "panda_joint6": 1.38774,
            "panda_joint7": 0.0,
            "panda_finger_joint2": 0.04,
        },
        pos=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
    # Stiffness and dampness of the panda arm parts
    # will be set
    actuators={
        "panda_arm1": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit=87,
            # velocity_limit=124.6,
        ),
        "panda_arm2": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            stiffness=0.0,
            damping=0.0,
            friction=0.0,
            armature=0.0,
            effort_limit=12,
            # velocity_limit=149.5,
        ),
        # Stiffness and dampness should be zero in order for these to not move
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint[1-2]"],
            effort_limit=40.0,
            # velocity_limit=0.04,
            stiffness=7500.0,
            damping=173.0,
            friction=0.1,
            armature=0.0,
        ),
    },
)

# Table
TABLE_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Table",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Mounts/UWPatVention/pat_vention.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.0, -0.868), rot=(0.70711, 0.0, 0.0, -0.70711)),
)

# NIST Board
NISTBOARD_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/NistBoard",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Props/NIST/Taskboard/nistboard.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.00)),
)

##
# Assembly Tools
##


BOLT_M16_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/BoltAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/factory_bolt_m16.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.55, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

NUT_M16_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/NutAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/factory_nut_m16.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.03),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.3, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)


HOLE_8MM_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/HoleAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/factory_hole_8mm.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.6, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)


PEG_8MM_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/PegAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/factory_peg_8mm.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.35, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)

SMALL_GEAR_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/SmallGearAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/factory_gear_small.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.4, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)


LARGE_GEAR_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/LargeGearAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/factory_gear_large.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.019),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.45, 0.1), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)


GEAR_BASE_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/GearBaseAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/factory_gear_base.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.65, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)


MEDIUM_GEAR_CFG: ArticulationCfg = ArticulationCfg(
    prim_path="/World/envs/env_.*/MediumGearAsset",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ASSET_DIR}/factory_gear_medium.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3666.0,
            enable_gyroscopic_forces=True,
            solver_position_iteration_count=192,
            solver_velocity_iteration_count=1,
            max_contact_impulse=1e32,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.012),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.4, 0.40, 0.0), rot=(1.0, 0.0, 0.0, 0.0), joint_pos={}, joint_vel={}
    ),
    actuators={},
)
