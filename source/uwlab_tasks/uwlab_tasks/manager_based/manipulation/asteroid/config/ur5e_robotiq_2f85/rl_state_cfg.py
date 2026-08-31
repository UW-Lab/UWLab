# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""State-based RL environments for the pick-only cube task (ASTEROID expert training).

Everything is derived from the OmniReset state environment; this module only expresses the
pick-specific deltas:

* no receptive object (scene, observations, events),
* pick-height + gripper-down success instead of assembly alignment,
* coupled sysid / OSC-gain / action-scale domain randomization during training,
* reset states loaded from the local ASTEROID dataset directory.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR
from uwlab_assets.robots.ur5e_robotiq_gripper import EXPLICIT_UR5E_ROBOTIQ_2F85

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.rl_state_cfg import (
    BaseEventCfg,
    FinetuneCurriculumsCfg,
    FinetuneEvalEventCfg,
    FinetuneEventCfg,
    ObservationsCfg,
    RewardsCfg,
    RlStateSceneCfg,
    Ur5eRobotiq2f85RlStateCfg,
)

from ... import ASTEROID_DATASETS_DIR
from ... import mdp as task_mdp
from .actions import Ur5eRobotiq2f85RelativeOSCAction, Ur5eRobotiq2f85RelativeOSCEvalAction

##
# Scene
##


def make_insertive_object(usd_path: str) -> RigidObjectCfg:
    """Insertive object with a stiffer solver than OmniReset's (16/2 vs 4/0 iterations)."""
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/InsertiveObject",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            scale=(1, 1, 1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=2,
                disable_gravity=False,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )


INSERTIVE_OBJECT_VARIANTS = {
    "fbleg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/SquareLeg/square_leg.usd"),
    "fbdrawerbottom": make_insertive_object(
        f"{UWLAB_CLOUD_ASSETS_DIR}/Props/FurnitureBench/DrawerBottom/drawer_bottom.usd"
    ),
    "peg": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Peg/peg.usd"),
    "cupcake": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/CupCake/cupcake.usd"),
    "cube": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd"),
    "rectangle": make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/Rectangle/rectangle.usd"),
}

variants = {"scene.insertive_object": INSERTIVE_OBJECT_VARIANTS}


@configclass
class PickSceneCfg(RlStateSceneCfg):
    """OmniReset state scene without the receptive object; cube by default."""

    receptive_object = None
    insertive_object = make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd")


##
# Events
##

RESET_SUCCESS_EXPR = "env.reward_manager.get_term_cfg('progress_context').func.success"

_TRAIN_RESET_TERM = dict(
    func=task_mdp.SingleObjectMultiResetManager,
    mode="reset",
    params={
        "dataset_dir": ASTEROID_DATASETS_DIR,
        "reset_types": ["ObjectAnywhereEEAnywhere", "ObjectRestingEEGrasped", "ObjectAnywhereEEGrasped"],
        "probs": [0.34, 0.33, 0.33],
        "success": RESET_SUCCESS_EXPR,
    },
)

_EVAL_RESET_TERM = dict(
    func=task_mdp.SingleObjectMultiResetManager,
    mode="reset",
    params={
        "dataset_dir": ASTEROID_DATASETS_DIR,
        "reset_types": ["ObjectAnywhereEEAnywhere"],
        "probs": [1.0],
        "success": RESET_SUCCESS_EXPR,
    },
)

_INSERTIVE_OBJECT_MASS_TERM = dict(
    func=task_mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("insertive_object"),
        # cube-sized objects: 20g - 100g
        "mass_distribution_params": (0.02, 0.1),
        "operation": "abs",
        "distribution": "uniform",
        "recompute_inertia": True,
    },
)

_UNIFIED_DR_TERM = dict(
    func=task_mdp.randomize_env_cfg_unified,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "joint_names": [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
        "actuator_name": "arm",
        "action_name": "arm",
        "arm_scale_range": (0.8, 1.2),
        "delay_range": (0, 1),
        "kp_scale_range": (0.8, 1.2),
        "terminal_kp": (1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
        "terminal_damping_ratio": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "initial_scales": (0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
        "target_scales": (0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
        "coupled_progress_range": (0.0, 1.5),
        "action_scale_progress_range": (0.0, 1.5),
    },
)


@configclass
class PickBaseEventCfg(BaseEventCfg):
    """OmniReset base events minus the receptive object; lighter insertive object."""

    receptive_object_material = None
    randomize_receptive_object_mass = None
    randomize_insertive_object_mass = EventTerm(**_INSERTIVE_OBJECT_MASS_TERM)


@configclass
class PickTrainEventCfg(PickBaseEventCfg):
    """Training events: 3-path resets + coupled sysid / gain / action-scale randomization."""

    reset_from_reset_states = EventTerm(**_TRAIN_RESET_TERM)
    randomize_env_cfg_unified = EventTerm(**_UNIFIED_DR_TERM)


@configclass
class PickTrainEvalEventCfg(PickBaseEventCfg):
    """Eval after Stage 1: no sysid / OSC gain randomization, 1-path resets."""

    reset_from_reset_states = EventTerm(**_EVAL_RESET_TERM)


@configclass
class PickFinetuneEventCfg(FinetuneEventCfg):
    """Finetune events: OmniReset's curriculum-ramped sysid + OSC gains, 3-path resets, unified DR."""

    receptive_object_material = None
    randomize_receptive_object_mass = None
    randomize_insertive_object_mass = EventTerm(**_INSERTIVE_OBJECT_MASS_TERM)
    reset_from_reset_states = EventTerm(**_TRAIN_RESET_TERM)
    randomize_env_cfg_unified = EventTerm(**_UNIFIED_DR_TERM)


@configclass
class PickFinetuneEvalEventCfg(FinetuneEvalEventCfg):
    """Eval after Stage 2 / data collection: fixed sysid + OSC gains, 1-path resets."""

    receptive_object_material = None
    randomize_receptive_object_mass = None
    randomize_insertive_object_mass = EventTerm(**_INSERTIVE_OBJECT_MASS_TERM)
    reset_from_reset_states = EventTerm(**_EVAL_RESET_TERM)


##
# Commands / observations / rewards
##


@configclass
class PickCommandsCfg:
    """Command specifications for the MDP."""

    task_command = task_mdp.PickTaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
    )


@configclass
class PickObservationsCfg(ObservationsCfg):
    """OmniReset observations minus every receptive-object term."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        receptive_asset_pose = None
        insertive_asset_in_receptive_asset_frame = None

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        receptive_asset_pose = None
        insertive_asset_in_receptive_asset_frame = None
        receptive_object_material_properties = None
        receptive_object_mass = None

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class PickRewardsCfg(RewardsCfg):
    """Pick-height success in place of assembly alignment; softer action / joint-velocity penalties."""

    action_rate = RewTerm(func=task_mdp.action_rate_l2_clamped, weight=-1e-4)

    joint_vel = RewTerm(
        func=task_mdp.joint_vel_l2_clamped,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
    )

    progress_context = RewTerm(
        func=task_mdp.ProgressContextPickOnly,  # type: ignore
        weight=0.1,
        params={
            "insertive_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_asset_cfg": SceneEntityCfg("robot", body_names="robotiq_base_link"),
            # success additionally requires the gripper pointing vertically down (approach axis
            # aligned with world -z within ~25 deg). Raise toward 1.0 to demand a stricter vertical.
            "gripper_down_dot_threshold": 0.9,
        },
    )

    dense_success_reward = RewTerm(func=task_mdp.dense_success_reward_pick_only, weight=0.1, params={"std": 1.0})

    success_reward = RewTerm(func=task_mdp.success_reward_pick_only, weight=1.0)


##
# Environments
##


@configclass
class Ur5eRobotiq2f85PickStateCfg(Ur5eRobotiq2f85RlStateCfg):
    """Base pick-only state environment (events set by the Train / Finetune / Eval subclasses)."""

    scene: PickSceneCfg = PickSceneCfg(num_envs=32, env_spacing=1.5)
    observations: PickObservationsCfg = PickObservationsCfg()
    rewards: PickRewardsCfg = PickRewardsCfg()
    commands: PickCommandsCfg = PickCommandsCfg()
    variants = variants

    def __post_init__(self):
        super().__post_init__()

        # Render settings: plain rasterization is enough for state-based training and keeps the
        # tactile data-collection cameras cheap.
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = False
        self.sim.render.enable_reflections = False
        self.sim.render.enable_dl_denoiser = False
        self.sim.render.antialiasing_mode = "DLAA"


# Training configuration (Stage 1: implicit actuator, coupled DR, no curriculum)
@configclass
class Ur5eRobotiq2f85PickRelCartesianOSCTrainCfg(Ur5eRobotiq2f85PickStateCfg):
    events: PickTrainEventCfg = PickTrainEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Finetune configuration (Stage 2: explicit actuator, curriculum ramps sysid + gains + scales)
@configclass
class Ur5eRobotiq2f85PickRelCartesianOSCFinetuneCfg(Ur5eRobotiq2f85PickStateCfg):
    """Finetune config: loads converged Stage 1 policy, explicit actuator from start, curriculum ramps DR."""

    events: PickFinetuneEventCfg = PickFinetuneEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    curriculum: FinetuneCurriculumsCfg = FinetuneCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")


# Evaluation configuration (after Stage 1: implicit actuator, soft gains, no sysid DR)
@configclass
class Ur5eRobotiq2f85PickRelCartesianOSCEvalCfg(Ur5eRobotiq2f85PickStateCfg):
    """Eval after Stage 1: implicit actuator, soft gains, large action scale, no sysid DR."""

    events: PickTrainEvalEventCfg = PickTrainEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


# Evaluation configuration (after Stage 2: explicit actuator, stiff gains, fixed sysid)
@configclass
class Ur5eRobotiq2f85PickRelCartesianOSCFinetuneEvalCfg(Ur5eRobotiq2f85PickStateCfg):
    """Eval after Stage 2: explicit actuator, stiff gains, small action scale, fixed sysid + OSC gains."""

    events: PickFinetuneEvalEventCfg = PickFinetuneEvalEventCfg()
    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = EXPLICIT_UR5E_ROBOTIQ_2F85.replace(prim_path="{ENV_REGEX_NS}/Robot")
