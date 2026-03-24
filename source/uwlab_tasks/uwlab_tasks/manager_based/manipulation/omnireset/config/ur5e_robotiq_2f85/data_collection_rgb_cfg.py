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
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from uwlab_assets import UWLAB_CLOUD_ASSETS_DIR

from ... import mdp as task_mdp
from .actions import Ur5eRobotiq2f85RelativeOSCEvalAction
from .rl_state_cfg import FinetuneEvalEventCfg, RlStateSceneCfg, Ur5eRobotiq2f85RlStateCfg


@configclass
class DataCollectionRGBObjectSceneCfg(RlStateSceneCfg):
    # background
    curtain_left = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainLeft",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, -0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        ),
    )

    curtain_back = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainBack",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-0.15, 0.0, 0.519), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.3, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        ),
    )

    curtain_right = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CurtainRight",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.4, 0.68, 0.519), rot=(0.707, 0.0, 0.0, -0.707)),
        spawn=sim_utils.CuboidCfg(
            size=(0.01, 1.0, 1.125),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=False,
            ),
        ),
    )

    front_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_front_camera",
        update_period=0,
        height=240,
        width=320,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(1.0770121, -0.1679045, 0.4486344),
            rot=(0.70564552, 0.46613815, 0.25072644, 0.47107948),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=13.20),
    )

    side_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/rgb_side_camera",
        update_period=0,
        height=240,
        width=320,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.8323904, 0.5877843, 0.2805111),
            rot=(0.29008842, 0.22122445, 0.51336143, 0.77676798),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=20.10),
    )

    wrist_camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/robotiq_base_link/rgb_wrist_camera",
        update_period=0,
        height=240,
        width=320,
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0182505, -0.00408447, -0.0689107),
            rot=(0.34254336, -0.61819255, -0.6160212, 0.347879),
            convention="opengl",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.55),
    )


@configclass
class BaseRGBEventCfg(FinetuneEvalEventCfg):
    """RGB events: inherits fixed sysid + OSC gains from FinetuneEvalEventCfg, adds camera randomization."""

    # randomize camera pose
    randomize_front_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/rgb_front_camera",
            # Base values from TiledCameraCfg
            "base_position": (1.0770121, -0.1679045, 0.4486344),
            "base_rotation": (0.70564552, 0.46613815, 0.25072644, 0.47107948),
            # Delta ranges for position (in meters)
            "position_deltas": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            # Delta ranges for euler angles (in degrees)
            "euler_deltas": {"pitch": (-2.0, 2.0), "yaw": (-2.0, 2.0), "roll": (-2.0, 2.0)},
        },
    )

    randomize_front_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/rgb_front_camera",
            "focal_length_range": (11.2, 15.2),
        },
    )

    randomize_side_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/rgb_side_camera",
            # Base values from TiledCameraCfg
            "base_position": (0.8323904, 0.5877843, 0.2805111),
            "base_rotation": (0.29008842, 0.22122445, 0.51336143, 0.77676798),
            # Delta ranges for position (in meters)
            "position_deltas": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
            # Delta ranges for euler angles (in degrees)
            "euler_deltas": {"pitch": (-2.0, 2.0), "yaw": (-2.0, 2.0), "roll": (-2.0, 2.0)},
        },
    )

    randomize_side_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={"camera_path_template": "/World/envs/env_{}/Robot/rgb_side_camera", "focal_length_range": (18.1, 22.1)},
    )

    randomize_wrist_camera = EventTerm(
        func=task_mdp.randomize_tiled_cameras,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/robotiq_base_link/rgb_wrist_camera",
            # Base values from TiledCameraCfg
            "base_position": (0.0182505, -0.00408447, -0.0689107),
            "base_rotation": (0.34254336, -0.61819255, -0.6160212, 0.347879),
            # Delta ranges for position (in meters)
            "position_deltas": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
            # Delta ranges for euler angles (in degrees)
            "euler_deltas": {"pitch": (-1.0, 1.0), "yaw": (-1.0, 1.0), "roll": (-1.0, 1.0)},
        },
    )

    randomize_wrist_camera_focal_length = EventTerm(
        func=task_mdp.randomize_camera_focal_length,
        mode="reset",
        params={
            "camera_path_template": "/World/envs/env_{}/Robot/robotiq_base_link/rgb_wrist_camera",
            "focal_length_range": (23.55, 25.55),  # Range from wide-angle to telephoto
        },
    )


@configclass
class RGBEventCfg(BaseRGBEventCfg):
    """Configuration for randomization."""

    randomize_wrist_mount_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "event_name": "randomize_wrist_mount_event",
            "mesh_names": ["robotiq_base_link/visuals/D415_to_Robotiq_Mount"],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.2, 1.0),
            "metallic_range": (0.0, 0.8),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_inner_finger_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "event_name": "randomize_inner_finger_event",
            "mesh_names": ["left_inner_finger/visuals/mesh_1", "right_inner_finger/visuals/mesh_1"],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.2, 1.0),
            "metallic_range": (0.0, 0.8),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_insertive_object_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "event_name": "randomize_insertive_object_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_receptive_object_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "event_name": "randomize_receptive_object_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_table_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "event_name": "randomize_table_event",
            "mesh_names": ["visuals/vention_mat"],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.3, 0.9),
            "metallic_range": (0.0, 0.3),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_curtain_left_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_left"),
            "event_name": "randomize_curtain_left_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_curtain_back_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_back"),
            "event_name": "randomize_curtain_back_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    randomize_curtain_right_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_right"),
            "event_name": "randomize_curtain_right_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
            "texture_scale_range": (0.7, 5.0),
            "roughness_range": (0.0, 1.0),
            "metallic_range": (0.0, 1.0),
            "specular_range": (0.0, 1.0),
        },
    )

    # reset background
    randomize_sky_light = EventTerm(
        func=task_mdp.randomize_hdri,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "light_path": "/World/skyLight",
            "hdri_config_path": str(Path(__file__).parent / "resources" / "hdri_paths.yaml"),
            "intensity_range": (1000.0, 4000.0),
            "rotation_range": (0.0, 360.0),
        },
    )

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class DataCollectionRGBEventCfg(RGBEventCfg):
    """Data collection events: override reset to sample from all 4 distributions."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class RGBCommandsCfg:
    """Command specifications for the MDP."""

    task_command = task_mdp.TaskCommandCfg(
        asset_cfg=SceneEntityCfg("robot", body_names="body"),
        resampling_time_range=(1e6, 1e6),
        insertive_asset_cfg=SceneEntityCfg("insertive_object"),
        receptive_asset_cfg=SceneEntityCfg("receptive_object"),
    )


@configclass
class RGBObservationsCfg:
    @configclass
    class RGBPolicyCfg(ObsGroup):
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

        front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224),
            },
        )

        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224),
            },
        )

        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                "process_image": True,
                "output_size": (224, 224),
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    @configclass
    class RGBDataCollectionCfg(ObsGroup):
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

        front_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("front_camera"),
                "data_type": "rgb",
                # Don't process image since we want save it as int8
                "process_image": False,
                "output_size": (224, 224),
            },
        )

        side_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("side_camera"),
                "data_type": "rgb",
                # Don't process image since we want save it as int8
                "process_image": False,
                "output_size": (224, 224),
            },
        )

        wrist_rgb = ObsTerm(
            func=task_mdp.process_image,
            params={
                "sensor_cfg": SceneEntityCfg("wrist_camera"),
                "data_type": "rgb",
                # Don't process image since we want save it as int8
                "process_image": False,
                "output_size": (224, 224),
            },
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

        insertive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

        receptive_asset_pose = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("receptive_object"),
                "root_asset_cfg": SceneEntityCfg("robot", body_names="wrist_3_link"),
                "rotation_repr": "axis_angle",
            },
        )

        insertive_asset_in_receptive_asset_frame: ObsTerm = ObsTerm(
            func=task_mdp.target_asset_pose_in_root_asset_frame,
            params={
                "target_asset_cfg": SceneEntityCfg("insertive_object"),
                "root_asset_cfg": SceneEntityCfg("receptive_object"),
                "rotation_repr": "axis_angle",
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: RGBPolicyCfg = RGBPolicyCfg()
    data_collection: RGBDataCollectionCfg = RGBDataCollectionCfg()


@configclass
class DataCollectionRGBTerminationsCfg:

    time_out = DoneTerm(func=task_mdp.time_out, time_out=True)

    abnormal_robot = DoneTerm(func=task_mdp.abnormal_robot_state)

    corrupted_camera = DoneTerm(
        func=task_mdp.corrupted_camera_detected,
        params={"camera_names": ["front_camera", "side_camera", "wrist_camera"], "std_threshold": 10.0},
    )

    early_success = DoneTerm(
        func=task_mdp.early_success_termination, params={"num_consecutive_successes": 5, "min_episode_length": 10}
    )

    success = DoneTerm(
        func=task_mdp.consecutive_success_state_with_min_length,
        params={"num_consecutive_successes": 5, "min_episode_length": 10},
    )


@configclass
class Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg(Ur5eRobotiq2f85RlStateCfg):
    """RGB base config: fixed sysid + RGB scene/obs/terminations/render."""

    actions: Ur5eRobotiq2f85RelativeOSCEvalAction = Ur5eRobotiq2f85RelativeOSCEvalAction()
    scene: DataCollectionRGBObjectSceneCfg = DataCollectionRGBObjectSceneCfg(
        num_envs=32, env_spacing=1.5, replicate_physics=False
    )
    observations: RGBObservationsCfg = RGBObservationsCfg()
    terminations: DataCollectionRGBTerminationsCfg = DataCollectionRGBTerminationsCfg()
    commands: RGBCommandsCfg = RGBCommandsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 32.0

        # Render settings
        self.sim.render.enable_dlssg = False
        self.sim.render.enable_ambient_occlusion = True
        self.sim.render.enable_reflections = True
        self.sim.render.enable_dl_denoiser = True
        self.sim.render.antialiasing_mode = "DLAA"

        # speeds up rendering
        self.sim.render_interval = self.decimation

        # rerender on reset
        self.num_rerenders_on_reset = 1


@configclass
class Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg(Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg):
    events: DataCollectionRGBEventCfg = DataCollectionRGBEventCfg()


@configclass
class Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg(Ur5eRobotiq2f85RGBRelCartesianOSCEvalCfg):
    """Evaluation config for Cartesian OSC delta actions."""

    events: RGBEventCfg = RGBEventCfg()


# OOD RGB Event Configs #
@configclass
class OODRGBEventCfg(BaseRGBEventCfg):
    """Configuration for randomization with OOD (out-of-distribution) textures and HDRIs."""

    # Override visual appearance randomization to use OOD textures
    randomize_wrist_mount_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "event_name": "randomize_wrist_mount_event",
            "mesh_names": ["robotiq_base_link/visuals/D415_to_Robotiq_Mount"],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths_ood.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_inner_finger_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "event_name": "randomize_inner_finger_event",
            "mesh_names": ["left_inner_finger/visuals/mesh_1", "right_inner_finger/visuals/mesh_1"],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths_ood.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_insertive_object_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("insertive_object"),
            "event_name": "randomize_insertive_object_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths_ood.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_receptive_object_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("receptive_object"),
            "event_name": "randomize_receptive_object_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths_ood.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_table_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "event_name": "randomize_table_event",
            "mesh_names": ["visuals/vention_mat"],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths_ood.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_curtain_left_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_left"),
            "event_name": "randomize_curtain_left_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths_ood.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_curtain_back_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_back"),
            "event_name": "randomize_curtain_back_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths_ood.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    randomize_curtain_right_appearance = EventTerm(
        func=task_mdp.randomize_visual_appearance_multiple_meshes,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "asset_cfg": SceneEntityCfg("curtain_right"),
            "event_name": "randomize_curtain_right_event",
            "mesh_names": [],
            "texture_prob": 0.5,
            "texture_config_path": str(Path(__file__).parent / "resources" / "texture_paths_ood.yaml"),
            "diffuse_tint_range": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
            "colors": {"r": (0.0, 1.0), "g": (0.0, 1.0), "b": (0.0, 1.0)},
        },
    )

    # Override HDRI randomization to use OOD HDRIs
    randomize_sky_light = EventTerm(
        func=task_mdp.randomize_hdri,
        mode="interval",
        interval_range_s=(4.0, 4.0),
        params={
            "light_path": "/World/skyLight",
            "hdri_config_path": str(Path(__file__).parent / "resources" / "hdri_paths_ood.yaml"),
            "intensity_range": (1000.0, 4000.0),
            "rotation_range": (0.0, 360.0),
        },
    )

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": ["ObjectAnywhereEEAnywhere"],
            "probs": [1.0],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class DataCollectionOODRGBEventCfg(OODRGBEventCfg):
    """Data collection OOD events: override reset to sample from all 4 distributions."""

    reset_from_reset_states = EventTerm(
        func=task_mdp.MultiResetManager,
        mode="reset",
        params={
            "dataset_dir": f"{UWLAB_CLOUD_ASSETS_DIR}/Datasets/OmniReset",
            "reset_types": [
                "ObjectAnywhereEEAnywhere",
                "ObjectRestingEEGrasped",
                "ObjectAnywhereEEGrasped",
                "ObjectPartiallyAssembledEEGrasped",
            ],
            "probs": [0.25, 0.25, 0.25, 0.25],
            "success": "env.reward_manager.get_term_cfg('progress_context').func.success",
        },
    )


@configclass
class Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCOODCfg(Ur5eRobotiq2f85DataCollectionRGBRelCartesianOSCCfg):
    """Data collection config with OOD (out-of-distribution) textures and HDRIs."""

    events: DataCollectionOODRGBEventCfg = DataCollectionOODRGBEventCfg()


@configclass
class Ur5eRobotiq2f85EvalRGBRelCartesianOSCOODCfg(Ur5eRobotiq2f85EvalRGBRelCartesianOSCCfg):
    """Evaluation config with OOD (out-of-distribution) textures and HDRIs."""

    events: OODRGBEventCfg = OODRGBEventCfg()
