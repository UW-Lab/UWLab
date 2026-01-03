# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from typing import Literal

from isaaclab.devices import DeviceBase
from isaaclab.utils import configclass

from .realsense_t265 import RealsenseT265
from .rokoko_glove import RokokoGlove
from .se3_keyboard import Se3Keyboard


@configclass
class DeviceBaseTeleopCfg:
    class_type: Callable[..., DeviceBase] = DeviceBase


@configclass
class KeyboardCfg(DeviceBaseTeleopCfg):
    class_type: Callable[..., Se3Keyboard] = Se3Keyboard

    pos_sensitivity: float = 0.01

    rot_sensitivity: float = 0.01

    enable_gripper_command: bool = False


@configclass
class RokokoGlovesCfg(DeviceBaseTeleopCfg):
    class_type: Callable[..., RokokoGlove] = RokokoGlove

    UDP_IP: str = "0.0.0.0"  # Listen on all available network interfaces

    UDP_PORT: int = 14043  # Make sure this matches the port used in Rokoko Studio Live

    left_hand_track: list[str] = []

    right_hand_track: list[str] = []

    scale: float = 1

    proximal_offset: float = 0.3

    thumb_scale: float = 1.1

    command_type: Literal["pose", "pos"] = "pos"


@configclass
class RealsenseT265Cfg(DeviceBaseTeleopCfg):
    class_type: Callable[..., RealsenseT265] = RealsenseT265

    cam_device_id: str = "905312110639"

    device: str = "cuda:0"
