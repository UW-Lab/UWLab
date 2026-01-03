# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

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
