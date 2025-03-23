# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Callable, Literal

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from uwlab.devices import DeviceBaseTeleopCfg

from .teleop import Teleop


@configclass
class TeleopCfg:
    @configclass
    class TeleopDevicesCfg:
        teleop_interface_cfg: DeviceBaseTeleopCfg = MISSING

        debug_vis: bool = False

        attach_body: SceneEntityCfg = MISSING

        attach_scope: Literal["self", "descendants"] = "self"

        command_type: Literal["position", "pose"] = MISSING

        pose_reference_body: SceneEntityCfg = MISSING

        reference_axis_remap: tuple[str, str, str] = MISSING

    class_type: Callable[..., Teleop] = Teleop

    teleop_devices: dict[str, TeleopDevicesCfg] = {}
