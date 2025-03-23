# Copyright (c) 2024-2025, The UW Lab Project Developers.
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
