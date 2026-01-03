# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators.actuator_base import ActuatorBase
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .actuator_cfg import EffortMotorCfg


class EffortMotor(ActuatorBase):
    cfg: EffortMotorCfg

    def __init__(self, cfg: EffortMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)

    def compute(self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor):
        self.computed_effort = torch.clip(control_action.joint_efforts, -self.effort_limit, self.effort_limit)
        self.applied_effort = self.computed_effort
        control_action.joint_efforts = self.applied_effort
        control_action.joint_positions = None
        control_action.joint_velocities = None
        return control_action

    def reset(self, env_ids: Sequence[int]):
        pass
