# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg, TiledCameraCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp
from .actions import Ur5eRobotiq2f85RelativeOSCEvalAction
from .rl_state_cfg import FinetuneEvalEventCfg, RlStateSceneCfg, Ur5eRobotiq2f85RlStateCfg


@configclass
class DataCollectionTactileObjectSceneCfg(RlStateSceneCfg):
    pass
    # background
    # curtain_left = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/CurtainLeft",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.01, 1.0, 1.125),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=False,
    #         ),
    #     ),
    # )

    # curtain_back = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/CurtainBack",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.0, 0.519), rot=(1.0, 0.0, 0.0, 0.0)),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.01, 1.3, 1.125),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=False,
    #         ),
    #     ),
    # )

    # curtain_right = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/CurtainRight",
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
    #     spawn=sim_utils.CuboidCfg(
    #         size=(0.01, 1.0, 1.125),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=False,
    #         ),
    #     ),
    # )

    # front_camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
    #     update_period=0,
    #     height=240,
    #     width=320,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(1.0770121, -0.1679045, 0.4486344),
    #         rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
    #         convention="opengl",
    #     ),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
    # )

    # side_camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
    #     update_period=0,
    #     height=240,
    #     width=320,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.8323904, 0.5877843, 0.2805111),
    #         rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
    #         convention="opengl",
    #     ),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(focal_length=20.10),
    # )

    # wrist_camera = TiledCameraCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
    #     update_period=0,
    #     height=240,
    #     width=320,
    #     offset=TiledCameraCfg.OffsetCfg(
    #         pos=(0.0182505, -0.00408447, -0.0689107),
    #         rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
    #         convention="opengl",
    #     ),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(focal_length=24.55),
    # )

    # right_inner_finger_contact_sensor = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/right_inner_finger", update_period=0.0, history_length=6, debug_vis=False, filter_prim_paths_expr=["{ENV_REGEX_NS}/InsertiveObject"]
    # )

    # left_inner_finger_contact_sensor = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/left_inner_finger", update_period=0.0, history_length=6, debug_vis=False, filter_prim_paths_expr=["{ENV_REGEX_NS}/InsertiveObject"]
    # )
    #TODO: replace with force torque sensors    


class EvalTactileObjectSceneCfg(DataCollectionTactileObjectSceneCfg):
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
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=13.20
        )
    )

@configclass
class TactileEventCfg(FinetuneEvalEventCfg):
    """Tactile events: inherits fixed sysid + OSC gains from FinetuneEvalEventCfg, adds sensor randomization."""
    #TODO: Add randomize contact sensors

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"logs/reset_state_datasets",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


# @configclass
# class DataCollectionTactileEventCfg(TactileEventCfg):
#     """Data collection events: override reset to sample from all 4 distributions."""

#     reset_from_reset_states = EventTerm(
#         func=task_mdp.MultiResetManager,
#         mode="reset",
#         params={
#             "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
#             "reset_types": [
#                 "ObjectAnywhereEEAnywhere",
#                 "ObjectRestingEEGrasped",
#                 "ObjectAnywhereEEGrasped",
#                 # "ObjectPartiallyAssembledEEGrasped",
#             ],
#             "probs": [0.25, 0.25, 0.25, 0.25],
#             "probs"
#             "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
#         },
#     )


@configclass
class TactileCommandCfg:
    """Command specifications for the MDP."""

    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        # receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )


@configclass
class TactileObservationsCfg:
    @configclass
    class TactilePolicyCfg(ObsGroup):
        """Observations for policy group (with processed images for evaluation)."""

        last_gripper_action = ObsTerm(
            func=task_mdp.last_action,
            params={
                "action_name": "gripper",
            },
        )

        last_arm_action = ObsTerm(
            func=task_mdp.last_action,
            params={
                "action_name": "arm",
            },
        )

        arm_joint_pos = ObsTerm(
            func=task_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"]),
            },
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
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_inner_finger_knuckle_joint"],
                ),
            }
        )

        # right_inner_finger_contact_force = ObsTerm(
        #     func=task_mdp.fingertip_contact_force_b,
        #     params={
        #         "contact_sensor_name": "right_inner_finger_contact_sensor",
        #         "root_asset_cfg": SceneEntityCfg("robot"),
        #         "root_body_name": "robotiq_base_link",
        #     },
        # )

        # left_inner_finger_contact_force = ObsTerm(
        #     func=task_mdp.fingertip_contact_force_b,
        #     params={
        #         "contact_sensor_name": "left_inner_finger_contact_sensor",
        #         "root_asset_cfg": SceneEntityCfg("robot"),
        #         "root_body_name": "robotiq_base_link",
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class TactileDataCollectionCfg(ObsGroup):
        """Observations for data collection group (with unprocessed images for saving)."""

        last_gripper_action = ObsTerm(
            func=task_mdp.last_action,
            params={
                "action_name": "gripper",
            },
        )

        last_arm_action = ObsTerm(
            func=task_mdp.last_action,
            params={
                "action_name": "arm",
            },
        )

        arm_joint_pos = ObsTerm(
            func=task_mdp.joint_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder.*", "elbow.*", "wrist.*"]),
            },
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
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[".*_inner_finger_knuckle_joint"],
                ),
            }
        )

        # Additional observations
        binary_contact = ObsTerm(
            func=task_mdp.binary_force_contact,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "body_name": "wrist_3_link",
                "force_threshold": 25.0,
            },
        )

        # insertive_asset_pose = ObsTerm(
        #     func=task_mdp.target_asset_pose_in_root_asset_frame,
        #     params={
        #         "target_asset_cfg": SceneEntityCfg("insertive_object"),
        #         "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
        #         "rotation_repr": "axis_angle",
        #     },
        # )

        # receptive_asset_pose = ObsTerm(
        #     func=task_mdp.target_asset_pose_in_root_asset_frame,
        #     params={
        #         "target_asset_cfg": SceneEntityCfg("receptive_object"),
        #         "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
        #         "rotation_repr": "axis_angle",
        #     },
        # )

        # insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
        #     func=task_mdp.target_asset_pose_in_root_asset_frame,
        #     params={
        #         "target_asset_cfg": SceneEntityCfg("insertive_object"),
        #         "root_asset_cfg": SceneEntityCfg("receptive_object"),
        #         "rotation_repr": "axis_angle",
        #     },
        # )

        # right_inner_finger_contact_force = ObsTerm(
        #     func=task_mdp.fingertip_contact_force_b,
        #     params={
        #         "contact_sensor_name": "right_inner_finger_contact_sensor",
        #         "root_asset_cfg": SceneEntityCfg("robot"),
        #         "root_body_name": "robotiq_base_link",
        #     },
        # )

        # left_inner_finger_contact_force = ObsTerm(
        #     func=task_mdp.fingertip_contact_force_b,
        #     params={
        #         "contact_sensor_name": "left_inner_finger_contact_sensor",
        #         "root_asset_cfg": SceneEntityCfg("robot"),
        #         "root_body_name": "robotiq_base_link",
        #     },
        # )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: TactilePolicyCfg = TactilePolicyCfg()
    data_collection: TactileDataCollectionCfg = TactileDataCollectionCfg()


@configclass
class DataCollectionTactileTerminationsCfg:

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    # corrupted_camera = DoneTerm(
    #     func=task_mdp.corrupted_camera_detected,
    #     params={"camera_names": ["front_camera", "side_camera", "wrist_camera"], "std_threshold": 10.0},
    # )

    early_success = DoneTerm(
        func=task_mdp.early_success_termination, params={"num_consecutive_successes": 5, "min_episode_length": 10}
    )

    success = DoneTerm(
        func=task_mdp.consecutive_success_state_with_min_length,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )


@configclass
class Ur5eRobotiq2f85TactileRelCartesianOSCEvalCfg(Ur5eRobotiq2f85RlStateCfg):
    """Tactile base config: fixed sysid + Tactile scene/obs/terminations/render."""

    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()
    scene: DataCollectionTactileObjectSceneCfg = DataCollectionTactileObjectSceneCfg(
        num_envs=32, env_spacing=1.5, replicate_physics=False
    )
    observations: TactileObservationsCfg = TactileObservationsCfg()
    terminations: DataCollectionTactileTerminationsCfg = DataCollectionTactileTerminationsCfg()
    commands: TactileCommandCfg = TactileCommandCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 16.0

        # Render settings
        # self.sim.render.enable_dlssg = False
        # self.sim.render.enable_ambient_occlusion = True
        # self.sim.render.enable_reflections = True
        # self.sim.render.enable_dl_denoiser = True
        # self.sim.render.antialiasing_mode = "DLAA"

        # speeds up rendering
        self.sim.render_interval = self.decimation

        # rerender on reset
        self.num_rerenders_on_reset = 1


@configclass
class Ur5eRobotiq2f85DataCollectionTactileRelCartesianOSCCfg(Ur5eRobotiq2f85TactileRelCartesianOSCEvalCfg):
    events: TactileEventCfg = TactileEventCfg()


@configclass
class Ur5eRobotiq2f85EvalTactileRelCartesianOSCCfg(Ur5eRobotiq2f85TactileRelCartesianOSCEvalCfg):
    """Evaluation config for Cartesian OSC delta actions."""
    scene: EvalTactileObjectSceneCfg = EvalTactileObjectSceneCfg(num_envs=32, env_spacing=1.5, replicate_physics=False)
    events: TactileEventCfg = TactileEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (1080, 1920)
            },
        )