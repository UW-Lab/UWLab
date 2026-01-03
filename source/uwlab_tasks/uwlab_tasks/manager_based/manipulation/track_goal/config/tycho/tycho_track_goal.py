# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import uwlab_assets.robots.tycho as tycho
import uwlab_assets.robots.tycho.mdp as tycho_mdp

import uwlab_tasks.manager_based.manipulation.track_goal.mdp as mdp

from ... import track_goal_env


class SceneCfg(track_goal_env.SceneCfg):
    ee_frame = tycho.FRAME_EE
    robot = tycho.HEBI_IMPLICIT_ACTUATOR_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    frame_fixed_chop_tip = tycho.FRAME_FIXED_CHOP_TIP
    frame_free_chop_tip = tycho.FRAME_FREE_CHOP_TIP


@configclass
class TerminationsCfg(track_goal_env.TerminationsCfg):
    """Termination terms for the MDP."""

    robot_invalid_state = DoneTerm(func=mdp.invalid_state, params={"asset_cfg": SceneEntityCfg("robot")})

    robot_extremely_bad_posture = DoneTerm(
        func=tycho_mdp.terminate_extremely_bad_posture,
        params={"probability": 0.5, "robot_cfg": SceneEntityCfg("robot")},
    )


@configclass
class GoalTrackingTychoEnv(track_goal_env.TrackGoalEnv):
    scene: SceneCfg = SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.decimation = 4
        self.sim.dt = 0.02 / self.decimation
        self.commands.ee_pose.body_name = "static_chop_tip"
        self.commands.ee_pose.ranges.pos_x = (-0.45, 0.0)
        self.commands.ee_pose.ranges.pos_y = (-0.325, -0.1)
        self.commands.ee_pose.ranges.pos_z = (0.05, 0.3)
        self.commands.ee_pose.ranges.roll = (1.57, 1.57)
        self.commands.ee_pose.ranges.pitch = (3.14, 3.14)
        self.commands.ee_pose.ranges.yaw = (-1.0, 1.0)
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = "static_chop_tip"
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = "static_chop_tip"
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = "static_chop_tip"
        self.rewards.end_effector_orientation_tracking_fine_grained.params["asset_cfg"].body_names = "static_chop_tip"


@configclass
class GoalTrackingTychoIkdelta(GoalTrackingTychoEnv):
    actions: tycho.IkdeltaAction = tycho.IkdeltaAction()


@configclass
class GoalTrackingTychoIkabsolute(GoalTrackingTychoEnv):
    actions: tycho.IkabsoluteAction = tycho.IkabsoluteAction()


@configclass
class GoalTrackingTychoJointPosition(GoalTrackingTychoEnv):
    actions: tycho.JointPositionAction = tycho.JointPositionAction()
