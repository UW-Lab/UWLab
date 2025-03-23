# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from uwlab.devices import KeyboardCfg, TeleopCfg


@configclass
class XarmUfGripperTeleopCfg:
    keyboard: TeleopCfg = TeleopCfg(
        teleop_devices={
            "device1": TeleopCfg.TeleopDevicesCfg(
                attach_body=SceneEntityCfg("robot", body_names="link_tcp"),
                attach_scope="self",
                pose_reference_body=SceneEntityCfg("robot", body_names="link_base"),
                reference_axis_remap=("x", "y", "z"),
                command_type="pose",
                debug_vis=True,
                teleop_interface_cfg=KeyboardCfg(
                    pos_sensitivity=0.01,
                    rot_sensitivity=0.01,
                    enable_gripper_command=True,
                ),
            ),
        }
    )
