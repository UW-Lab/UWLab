# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from uwlab_assets.robots.leap.articulation_drive.dynamixel_driver_cfg import DynamixelDriverCfg
from uwlab_assets.robots.xarm.articulation_drive.xarm_driver_cfg import XarmDriverCfg

from uwlab.assets import ArticulationCfg
from uwlab.assets.articulation import BulletArticulationViewCfg
from uwlab.envs import RealRLEnvCfg
from uwlab.scene import SceneContextCfg

import uwlab_tasks.manager_based.manipulation.track_goal.mdp as mdp


@configclass
class SceneCfg(SceneContextCfg):
    robot = ArticulationCfg(
        articulation_view_cfg=BulletArticulationViewCfg(
            drive_cfg=XarmDriverCfg(
                ip="192.168.1.220",
                work_space_limit=[[0.2, 0.6], [-0.325, 0.325], [0.15, 0.5]],
                is_radian=True,
                p_gain_scaler=0.01,
            ),
            isaac_joint_names=["joint1", "joint2", "joint3", "joint4", "joint5"],
            urdf="/path/to/xarm5.urdf",
            dt=0.01,  # recommended hz is 30 - 300hz
            debug_visualize=False,
            # dummy_mode=True,
            use_multiprocessing=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={"joint1": 0.0, "joint3": -0.5}),
        actuators={"xarm": ImplicitActuatorCfg(joint_names_expr=["joint.*"], stiffness=0.0, damping=0.0)},
    )

    hand = ArticulationCfg(
        articulation_view_cfg=BulletArticulationViewCfg(
            drive_cfg=DynamixelDriverCfg(port="/dev/ttyUSB0"),
            urdf="/path/to/leap.urdf",
            # fmt: off
            isaac_joint_names=[
                'j1', 'j12', 'j5', 'j9', 'j0', 'j13', 'j4', 'j8', 'j2', 'j14', 'j6', 'j10', 'j3', 'j15', 'j7', 'j11'
            ],
            # fmt: on
            debug_visualize=False,
            use_multiprocessing=True,
            dt=0.01,
            # dummy_mode=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(joint_pos={"j.*": 0.0}),
        actuators={
            "mcp_joints": ImplicitActuatorCfg(joint_names_expr=["j[048]"], stiffness=50, damping=50),
            "other_joints": ImplicitActuatorCfg(joint_names_expr=["j(?![048]).*"], stiffness=80, damping=50),
        },
    )


@configclass
class ActionCfg:
    xarm_joint_position_action = JointPositionActionCfg(
        asset_name="robot", joint_names=["joint.*"], scale=1.0, use_default_offset=False
    )

    leap_hand_joint_position_action = JointPositionActionCfg(
        asset_name="hand", joint_names=["j.*"], scale=1.0, use_default_offset=False
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        xarm_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        hand_joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("hand")})

        xarm_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        hand_joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("hand")})

        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)
        # hand_target = ObsTerm(func=mdp.generated_commands, params={"command_name": "hand_posture"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="link_eef",
        resampling_time_range=(1.5, 2.5),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.3, 0.5),
            pos_y=(-0.225, 0.225),
            pos_z=(0.125, 0.4),
            roll=(0, 0.0),
            pitch=(0, 1),
            yaw=(0.0, 1),
        ),
    )


@configclass
class EventCfg:
    reset_robot_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": [0.0, 0.0],
            "velocity_range": [0.0, 0.0],
        },
        mode="reset",
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    pass
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l2,
    #     weight=-0.0001,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )

    # end_effector_position_tracking = RewTerm(
    #     func=mdp.link_position_command_align_tanh,
    #     weight=0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="REPLACE_ME"),
    #         "std": 0.5,
    #         "command_name": "ee_pose",
    #     },
    # )

    # end_effector_position_tracking_fine_grained = RewTerm(
    #     func=mdp.link_position_command_align_tanh,
    #     weight=1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="REPLACE_ME"),
    #         "std": 0.1,
    #         "command_name": "ee_pose",
    #     },
    # )

    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.link_orientation_command_align_tanh,
    #     weight=0.5,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="REPLACE_ME"),
    #         "std": 0.5,
    #         "command_name": "ee_pose",
    #     },
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class TrackGoalXarmLeapDeployment(RealRLEnvCfg):
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    scene: SceneCfg = SceneCfg()
    actions: ActionCfg = ActionCfg()
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 1
        self.episode_length_s = 50
        self.scene.dt = 0.02
