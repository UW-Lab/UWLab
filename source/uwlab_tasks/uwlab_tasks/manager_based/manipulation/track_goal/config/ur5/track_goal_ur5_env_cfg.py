# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import uwlab_assets.robots.ur5 as ur5

from ... import mdp, track_goal_env


@configclass
class Ur5SceneCfg(track_goal_env.SceneCfg):
    robot = ur5.IMPLICIT_UR5.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class Ur5TermincationCfg(track_goal_env.TerminationsCfg):
    robot_invalid_state = DoneTerm(func=mdp.abnormal_robot_state)


@configclass
class TrackGoalUr5EnvCfg(track_goal_env.TrackGoalEnv):
    scene: Ur5SceneCfg = Ur5SceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=False)
    terminations: Ur5TermincationCfg = Ur5TermincationCfg()

    def __post_init__(self):
        # simulation settings
        super().__post_init__()
        self.decimation = 4
        self.sim.dt = 0.02 / self.decimation
        self.viewer.eye = (3.0, 3.0, 1.0)

        # Contact and solver settings
        self.sim.physx.solver_type = 1
        # Render settings
        self.sim.render.enable_dlssg = True
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
        self.commands.ee_pose.body_name = "robotiq_base_link"
        self.commands.ee_pose.ranges.pos_x = (-0.5, -0.25)
        self.commands.ee_pose.ranges.pitch = (0.5, 1.57)
        self.commands.ee_pose.ranges.yaw = (1.57, 4.61)

        self.events.reset_robot_joint = None  # don't reset robotiq joints, it may cause robot to fail

        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = "robotiq_base_link"
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = "robotiq_base_link"
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = "robotiq_base_link"
        self.rewards.end_effector_orientation_tracking_fine_grained.params["asset_cfg"].body_names = "robotiq_base_link"
        self.rewards.end_effector_orientation_tracking.weight *= 2.0


@configclass
class TrackGoalUr5IkAbsEnvCfg(TrackGoalUr5EnvCfg):
    actions: ur5.Ur5IkAbsoluteAction = ur5.Ur5IkAbsoluteAction()


@configclass
class TrackGoalUr5JointPositionEnvCfg(TrackGoalUr5EnvCfg):
    actions: ur5.Ur5JointPositionAction = ur5.Ur5JointPositionAction()
