# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Proprioceptive ("tactile") data-collection and evaluation environments for ASTEROID.

The student policy sees only proprioception (arm joints, EE pose, last actions and a
normalized gripper-position reading with calibration-drift randomization); the state expert
that supervises it sees the pick-only policy observation group. Used by
``scripts/ASTEROID/collect_demos_asteroid.py`` and ``scripts/ASTEROID/eval_asteroid_policy.py``.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from ... import mdp as task_mdp
from .actions import Ur5eRobotiq2f85RelativeOSCAction, Ur5eRobotiq2f85RelativeOSCEvalAction
from .rl_state_cfg import PickFinetuneEvalEventCfg, PickObservationsCfg, PickSceneCfg, Ur5eRobotiq2f85PickStateCfg

##
# Scene
##


@configclass
class TactileSceneCfg(PickSceneCfg):
    """Pick scene for data collection (no cameras; the student is proprioceptive)."""

    # TODO: add fingertip force/torque sensors once the real-robot counterpart is available.


@configclass
class TactileEvalSceneCfg(TactileSceneCfg):
    """Adds a high-resolution front camera for evaluation videos."""

    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
        update_period=0,
        height=1080,
        width=1920,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.21290445, 0.4486344),
            rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
    )


##
# Events
##


@configclass
class TactileEventCfg(PickFinetuneEvalEventCfg):
    """Fixed sysid + OSC gains, 1-path resets, plus gripper-reading calibration randomization."""

    randomize_gripper_pos_affine = EventTerm(
        func=task_mdp.randomize_gripper_pos_affine,
        mode="reset",
        params={
            "scale_range": (0.9, 1.1),
            "offset_range": (-0.03, 0.03),
        },
    )


##
# Observations
##

_GRIPPER_POS_TERM = dict(
    func=task_mdp.gripper_pos_normalized,
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=["left_inner_finger_knuckle_joint"]),
        "scale_event_name": "randomize_gripper_pos_affine",
        "jitter_std": 0.01,
    },
)


@configclass
class TactileObservationsCfg:
    @configclass
    class TactilePolicyCfg(ObsGroup):
        """Student (diffusion) policy obs -- dict-form; keys must match the ``shape_meta`` of the
        diffusion-policy task config so the student receives the layout it was trained on."""

        last_gripper_action = ObsTerm(func=task_mdp.last_action, params={"action_name": "gripper"})

        last_arm_action = ObsTerm(func=task_mdp.last_action, params={"action_name": "arm"})

        arm_joint_pos = ObsTerm(
            func=task_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
        )

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        gripper_pos = ObsTerm(**_GRIPPER_POS_TERM)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class TactileDataCollectionCfg(ObsGroup):
        """Observations recorded to the dataset (policy obs + raw gripper joint angles)."""

        last_gripper_action = ObsTerm(func=task_mdp.last_action, params={"action_name": "gripper"})

        last_arm_action = ObsTerm(func=task_mdp.last_action, params={"action_name": "arm"})

        arm_joint_pos = ObsTerm(
            func=task_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"])},
        )

        end_effector_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "root_asset_cfg": SceneEntityCfg("robot"),
                "rotation_repr": "axis_angle",
            },
        )

        gripper_joint_pos = ObsTerm(
            func=task_mdp.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_inner_finger_knuckle_joint"])},
        )

        gripper_pos = ObsTerm(**_GRIPPER_POS_TERM)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: TactilePolicyCfg = TactilePolicyCfg()
    data_collection: TactileDataCollectionCfg = TactileDataCollectionCfg()
    # Privileged state observations consumed by the JIT-loaded state expert during BC
    # supervision -- identical to the group the expert was trained on. Read by
    # ``my_experts_observation_func`` via ``env.unwrapped.obs_buf["expert_obs"]``.
    expert_obs: PickObservationsCfg.PolicyCfg = PickObservationsCfg.PolicyCfg()


##
# Terminations
##


@configclass
class TactileTerminationsCfg:
    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    early_success = DoneTerm(
        func=task_mdp.early_success_termination, params={"num_consecutive_successes": 10, "min_episode_length": 10}
    )

    success = DoneTerm(
        func=task_mdp.consecutive_success_state_with_min_length,
        params={"num_consecutive_successes": 10, "min_episode_length": 10},
    )


##
# Environments
##


@configclass
class Ur5eRobotiq2f85TactileRelCartesianOSCEvalCfg(Ur5eRobotiq2f85PickStateCfg):
    """Tactile base config: fixed sysid + tactile scene / obs / terminations / render."""

    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()
    scene: TactileSceneCfg = TactileSceneCfg(num_envs=32, env_spacing=1.5, replicate_physics=False)
    observations: TactileObservationsCfg = TactileObservationsCfg()
    terminations: TactileTerminationsCfg = TactileTerminationsCfg()
    events: TactileEventCfg = TactileEventCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0

        # speeds up rendering
        self.sim.render_interval = self.decimation

        # rerender on reset
        self.num_rerenders_on_reset = 1


@configclass
class Ur5eRobotiq2f85DataCollectionTactileRelCartesianOSCCfg(Ur5eRobotiq2f85TactileRelCartesianOSCEvalCfg):
    """Data collection with the Stage 1 (soft-gain) expert."""

    actions: Ur5eRobotiq2f85RelativeOSCAction = Ur5eRobotiq2f85RelativeOSCAction()


@configclass
class Ur5eRobotiq2f85DataCollectionFinetuneTactileRelCartesianOSCCfg(Ur5eRobotiq2f85TactileRelCartesianOSCEvalCfg):
    """Data collection with the Stage 2 (stiff-gain, finetuned) expert."""

    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()


@configclass
class Ur5eRobotiq2f85EvalTactileRelCartesianOSCCfg(Ur5eRobotiq2f85TactileRelCartesianOSCEvalCfg):
    """Evaluation of a Stage 1 student, with a front camera for videos."""

    scene: TactileEvalSceneCfg = TactileEvalSceneCfg(num_envs=32, env_spacing=1.5, replicate_physics=False)

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (1080, 1920),
            },
        )


@configclass
class Ur5eRobotiq2f85EvalFinetuneTactileRelCartesianOSCCfg(Ur5eRobotiq2f85EvalTactileRelCartesianOSCCfg):
    """Evaluation of a Stage 2 student (stiff gains), with a front camera for videos."""

    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()
