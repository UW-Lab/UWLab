# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reset-state recording environments for the pick-only cube task.

Subclasses the OmniReset reset-state configs, removes the receptive object and narrows the
object / end-effector sampling regions to a cube resting on the table in front of the robot.
Recorded datasets go to ``<ASTEROID_DATASETS_DIR>/Resets/<Object>/`` (see
:mod:`uwlab_tasks.manager_based.manipulation.asteroid`).
"""

from __future__ import annotations

import numpy as np

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.reset_states_cfg import (
    ObjectAnywhereEEAnywhereEventCfg,
    ObjectAnywhereEEGraspedEventCfg,
    ObjectRestingEEGraspedEventCfg,
    ResetStatesSceneCfg,
    ResetStatesTerminationCfg,
    UR5eRobotiq2f85ResetStatesCfg,
    make_insertive_object,
)

from ... import ASTEROID_DATASETS_DIR
from ... import mdp as task_mdp

##
# Scene
##

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
class PickResetStatesSceneCfg(ResetStatesSceneCfg):
    """OmniReset reset-state scene without the receptive object; cube by default."""

    receptive_object = None
    insertive_object = make_insertive_object(f"{UWLAB_CLOUD_ASSETS_DIR}/Props/Custom/InsertiveCube/insertive_cube.usd")


##
# Events
##


@configclass
class PickObjectAnywhereEEAnywhereEventCfg(ObjectAnywhereEEAnywhereEventCfg):
    """Cube resting flat on the table in front of the robot; EE hovering above it, pointing down."""

    receptive_object_material = None
    reset_receptive_object_pose = None

    reset_insertive_object_pose = EventTerm(
        func=task_mdp.reset_root_states_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.4, 0.5),
                "y": (0.05, 0.15),
                "z": (0.01, 0.02),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (-np.pi / 8, np.pi / 8),
            },
            "velocity_range": {},
            "asset_cfgs": {"insertive_object": SceneEntityCfg("insertive_object")},
            "offset_asset_cfg": SceneEntityCfg("ur5_metal_support"),
            "use_bottom_offset": True,
        },
    )

    reset_end_effector_pose = EventTerm(
        func=task_mdp.reset_end_effector_round_fixed_asset,
        mode="reset",
        params={
            "fixed_asset_cfg": SceneEntityCfg("robot"),
            "fixed_asset_offset": None,
            "pose_range_b": {
                "x": (0.44, 0.46),
                "y": (0.09, 0.11),
                "z": (0.18, 0.2),
                "roll": (0.0, 0.0),
                "pitch": (np.pi / 2 - 0.1, np.pi / 2 + 0.1),
                "yaw": (np.pi - 0.1, np.pi + 0.1),
            },
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
        },
    )


@configclass
class PickObjectRestingEEGraspedEventCfg(ObjectRestingEEGraspedEventCfg):
    receptive_object_material = None
    reset_receptive_object_pose = None

    reset_insertive_object_pose_from_reset_states = EventTerm(
        func=task_mdp.SingleObjectMultiResetManager,
        mode="reset",
        params={
            "dataset_dir": ASTEROID_DATASETS_DIR,
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
        },
    )

    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "dataset_dir": ASTEROID_DATASETS_DIR,
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"]),
            "pose_range_b": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "z": (-0.02, 0.02),
                "roll": (-np.pi / 16, np.pi / 16),
                "pitch": (-np.pi / 16, np.pi / 16),
                "yaw": (-np.pi / 16, np.pi / 16),
            },
        },
    )


@configclass
class PickObjectAnywhereEEGraspedEventCfg(ObjectAnywhereEEGraspedEventCfg):
    receptive_object_material = None
    reset_receptive_object_pose = None

    reset_end_effector_pose_from_grasp_dataset = EventTerm(
        func=task_mdp.reset_end_effector_from_grasp_dataset,
        mode="reset",
        params={
            "dataset_dir": ASTEROID_DATASETS_DIR,
            "fixed_asset_cfg": SceneEntityCfg("insertive_object"),
            "robot_ik_cfg": SceneEntityCfg(
                "robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"], body_names="robotiq_base_link"
            ),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=["finger_joint", ".*right.*", ".*left.*"]),
            "pose_range_b": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


##
# Terminations
##


@configclass
class PickResetStatesTerminationCfg(ResetStatesTerminationCfg):
    """Reset-state validity check against the insertive object only."""

    success = DoneTerm(
        func=task_mdp.check_reset_state_success,
        params={
            "object_cfgs": [SceneEntityCfg("insertive_object")],
            "robot_cfg": SceneEntityCfg("robot"),
            "ee_body_name": "robotiq_base_link",
            "collision_analyzer_cfgs": [
                task_mdp.CollisionAnalyzerCfg(
                    num_points=1024,
                    max_dist=0.5,
                    min_dist=-0.0005,
                    asset_cfg=SceneEntityCfg("robot"),
                    obstacle_cfgs=[SceneEntityCfg("insertive_object")],
                ),
            ],
            "max_robot_pos_deviation": 0.1,
            "max_object_pos_deviation": np.inf,
            "pos_z_threshold": -0.02,
            "consecutive_stability_steps": 5,
        },
        time_out=True,
    )


##
# Environments
##


@configclass
class PickResetStatesCfg(UR5eRobotiq2f85ResetStatesCfg):
    """Base reset-state environment for the pick-only task."""

    scene: PickResetStatesSceneCfg = PickResetStatesSceneCfg(num_envs=1, env_spacing=1.5)
    terminations: PickResetStatesTerminationCfg = PickResetStatesTerminationCfg()
    variants = variants


@configclass
class PickObjectAnywhereEEAnywhereResetStatesCfg(PickResetStatesCfg):
    events: PickObjectAnywhereEEAnywhereEventCfg = PickObjectAnywhereEEAnywhereEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = np.inf


@configclass
class PickObjectRestingEEGraspedResetStatesCfg(PickResetStatesCfg):
    events: PickObjectRestingEEGraspedEventCfg = PickObjectRestingEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.01


@configclass
class PickObjectAnywhereEEGraspedResetStatesCfg(PickResetStatesCfg):
    events: PickObjectAnywhereEEGraspedEventCfg = PickObjectAnywhereEEGraspedEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.terminations.success.params["max_object_pos_deviation"] = 0.05
