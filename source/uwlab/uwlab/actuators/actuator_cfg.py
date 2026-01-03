# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.actuators.actuator_cfg import ActuatorBaseCfg
from isaaclab.utils import configclass

from . import actuator_pd


@configclass
class EffortMotorCfg(ActuatorBaseCfg):
    class_type: type = actuator_pd.EffortMotor
