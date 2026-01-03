# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from uwlab.devices import KeyboardCfg, RealsenseT265Cfg, RokokoGlovesCfg, TeleopCfg


@configclass
class LeapTeleopCfg:
    keyboard_rokokoglove: TeleopCfg = TeleopCfg(
        teleop_devices={
            "device1": TeleopCfg.TeleopDevicesCfg(
                attach_body=SceneEntityCfg("robot", body_names="wrist"),
                attach_scope="self",
                pose_reference_body=SceneEntityCfg("robot", body_names="link_base"),
                reference_axis_remap=("x", "y", "z"),
                command_type="pose",
                debug_vis=True,
                teleop_interface_cfg=KeyboardCfg(
                    pos_sensitivity=0.01,
                    rot_sensitivity=0.01,
                    enable_gripper_command=False,
                ),
            ),
            "device2": TeleopCfg.TeleopDevicesCfg(
                attach_body=SceneEntityCfg("robot", body_names="wrist"),
                attach_scope="descendants",
                pose_reference_body=SceneEntityCfg("robot", body_names="link_base"),
                reference_axis_remap=("x", "y", "z"),
                command_type="position",
                debug_vis=True,
                teleop_interface_cfg=RokokoGlovesCfg(
                    UDP_IP="0.0.0.0",
                    UDP_PORT=14043,
                    scale=1.55,
                    thumb_scale=0.9,
                    right_hand_track=[
                        "rightIndexMedial",
                        "rightMiddleMedial",
                        "rightRingMedial",
                        "rightIndexTip",
                        "rightMiddleTip",
                        "rightRingTip",
                        "rightThumbTip",
                    ],
                ),
            ),
        }
    )

    realsense_rokokoglove: TeleopCfg = TeleopCfg(
        teleop_devices={
            "device1": TeleopCfg.TeleopDevicesCfg(
                attach_body=SceneEntityCfg("robot", body_names="wrist"),
                attach_scope="self",
                pose_reference_body=SceneEntityCfg("robot", body_names="link_base"),
                reference_axis_remap=("x", "y", "z"),
                command_type="pose",
                debug_vis=True,
                teleop_interface_cfg=RealsenseT265Cfg(),
            ),
            "device2": TeleopCfg.TeleopDevicesCfg(
                attach_body=SceneEntityCfg("robot", body_names="wrist"),
                attach_scope="descendants",
                pose_reference_body=SceneEntityCfg("robot", body_names="link_base"),
                reference_axis_remap=("x", "y", "z"),
                command_type="position",
                debug_vis=True,
                teleop_interface_cfg=RokokoGlovesCfg(
                    UDP_IP="0.0.0.0",
                    UDP_PORT=14043,
                    scale=1.55,
                    thumb_scale=0.9,
                    right_hand_track=[
                        "rightIndexMedial",
                        "rightMiddleMedial",
                        "rightRingMedial",
                        "rightIndexTip",
                        "rightMiddleTip",
                        "rightRingTip",
                        "rightThumbTip",
                    ],
                ),
            ),
        }
    )
