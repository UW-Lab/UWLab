# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package providing interfaces to different teleoperation devices.

Currently, the following categories of devices are supported:

* **Se3Keyboard**: Standard keyboard with WASD and arrow keys.
* **RealsenseT265**: Realsense cameras 6 degrees of freedom.
* **RokokoGlove**: Rokoko gloves that track position and quaternion of the hand.

All device interfaces inherit from the :class:`isaaclab.devices.DeviceBase` class, which provides a
common interface for all devices. The device interface reads the input data when
the :meth:`DeviceBase.advance` method is called. It also provides the function :meth:`DeviceBase.add_callback`
to add user-defined callback functions to be called when a particular input is pressed from
the peripheral device.
"""

from .device_cfg import *
from .realsense_t265 import RealsenseT265
from .rokoko_glove import RokokoGlove
from .se3_keyboard import Se3Keyboard
from .teleop_cfg import TeleopCfg
