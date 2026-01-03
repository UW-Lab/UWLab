# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import uwlab_assets.robots.leap as leap
import uwlab_assets.robots.leap.mdp as leap_mdp
import uwlab_assets.robots.xarm_leap as xarm_leap
from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from uwlab.envs.mdp.actions import VisualizableJointTargetPositionCfg

import uwlab_tasks.manager_based.manipulation.track_goal.mdp as mdp

from ... import track_goal_env


@configclass
class SceneCfg(track_goal_env.SceneCfg):
    robot = xarm_leap.IMPLICIT_XARM_LEAP.replace(prim_path="{ENV_REGEX_NS}/Robot")

    hand_command_articulation_vis = leap.IMPLICIT_LEAP.replace(
        prim_path="{ENV_REGEX_NS}/HANDVIS",
        spawn=leap.IMPLICIT_LEAP.spawn.replace(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/LeapHand/leap_transparent.usd",
        ),
    )

    xarm_leap_action_vis = xarm_leap.IMPLICIT_XARM_LEAP.replace(
        prim_path="{ENV_REGEX_NS}/ROBOTACTIONVIZ",
        spawn=xarm_leap.IMPLICIT_XARM_LEAP.spawn.replace(
            usd_path=f"{UWLAB_CLOUD_ASSETS_DIR}/Robots/UFactory/Xarm5LeapHand/leap_xarm_visual.usd",
        ),
        init_state=xarm_leap.IMPLICIT_XARM_LEAP.init_state.replace(pos=(0.0, 0.0, 0.0)),
    )


@configclass
class EventCfg:
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )

    reset_robot_joint = EventTerm(
        func=mdp.reset_joints_by_scale,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.5, 0.5),
            "velocity_range": (-0.1, 0.1),
        },
        mode="reset",
    )


@configclass
class ObservationCfg(track_goal_env.ObservationsCfg):
    @configclass
    class PolicyCfg(track_goal_env.ObservationsCfg.PolicyCfg):
        """Observations for policy group."""

        hand_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_posture"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardCfg(track_goal_env.RewardsCfg):
    hand_joint_command_tracking = RewTerm(
        func=mdp.joint_position_command_error_l2_norm,
        weight=-1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names="j[0-9]+"),
            "command_name": "hand_posture",
        },
    )

    stay_still = RewTerm(
        func=mdp.stay_still,
        weight=-0.01,
        params={
            "ee_command_name": "ee_pose",
            "hand_command_name": "hand_posture",
            "hand_asset_cfg": SceneEntityCfg("robot", joint_names="j[0-9]+"),
            "ee_asset_cfg": SceneEntityCfg("robot", body_names="palm_lower"),
        },
    )

    delta_action_l2 = RewTerm(
        func=mdp.delta_action_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class CommandCfg(track_goal_env.CommandsCfg):
    hand_posture = mdp.HandJointCommandCfg(
        debug_vis=True,
        wrist_pose_term="ee_pose",
        asset_cfg=SceneEntityCfg("robot", joint_names=["j[0-9]+"]),
        articulation_vis_cfg=SceneEntityCfg("hand_command_articulation_vis", joint_names=["j[0-9]+"]),
        resampling_time_range=(1.5, 2.5),
        predefined_hand_joint_goals=[
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [1.25, 0.50, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.50, 0.75, 1.25, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [1.25, 0.50, 0.60, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.50, 0.75, 1.25, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 1.00, 1.00, 0.00, 0.00, 0.00, 0.00],
        ],
    )


@configclass
class TrackGoalXarmLeap(track_goal_env.TrackGoalEnv):
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    events: EventCfg = EventCfg()
    observations: ObservationCfg = ObservationCfg()
    commands: CommandCfg = CommandCfg()
    rewards: RewardCfg = RewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 5
        self.sim.dt = 0.02 / self.decimation
        self.commands.ee_pose.body_name = "palm_lower"
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = "palm_lower"
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = "palm_lower"
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = "palm_lower"
        self.rewards.end_effector_orientation_tracking_fine_grained.params["asset_cfg"].body_names = "palm_lower"
        self.rewards.action_rate.weight *= 5.0
        self.rewards.end_effector_orientation_tracking.weight *= 2.0
        self.rewards.end_effector_position_tracking.weight *= 1.5
        self.rewards.end_effector_position_tracking_fine_grained.weight *= 1.5
        if not self.commands.hand_posture.debug_vis:
            self.scene.hand_command_articulation_vis = None
        if not hasattr(self.actions, "jointpos_viz") or (self.actions.jointpos_viz.debug_vis is False):
            self.scene.xarm_leap_action_vis = None
        if self.commands.hand_posture.debug_vis or (
            hasattr(self.actions, "jointpos_viz") and self.actions.jointpos_viz.debug_vis
        ):
            # this is necessary to visualize opacity in the raytracing
            import carb
            import isaacsim.core.utils.carb as carb_utils

            self.sim.render.enable_translucency = True
            # # Access the Carb settings registry
            settings = carb.settings.get_settings()
            carb_utils.set_carb_setting(settings, "/rtx/raytracing/fractionalCutoutOpacity", True)


@configclass
class TargetVizableJointPositionAction(xarm_leap.XarmLeapJointPositionAction):
    jointpos_viz: VisualizableJointTargetPositionCfg = VisualizableJointTargetPositionCfg(
        debug_vis=False,
        asset_name="robot",
        joint_names=["joint.*", "j[0-9]+"],
        articulation_vis_cfg=SceneEntityCfg("xarm_leap_action_vis"),
    )

    leap_action_correction = leap_mdp.LeapJointPositionActionCorrectionCfg(
        asset_name="robot",
        joint_names=["joint.*", "j[0-9]+"],
    )


@configclass
class TrackGoalXarmLeapVizJointPosition(TrackGoalXarmLeap):
    actions: TargetVizableJointPositionAction = TargetVizableJointPositionAction()  # type: ignore


@configclass
class TrackGoalXarmLeapJointPosition(TrackGoalXarmLeap):
    actions = xarm_leap.XarmLeapJointPositionAction()  # type: ignore


@configclass
class TrackGoalXarmLeapMcIkAbs(TrackGoalXarmLeap):
    actions = xarm_leap.XarmLeapMcIkAbsoluteAction()  # type: ignore


@configclass
class TrackGoalXarmLeapMcIkDel(TrackGoalXarmLeap):
    actions = xarm_leap.XarmLeapMcIkDeltaAction()  # type: ignore
