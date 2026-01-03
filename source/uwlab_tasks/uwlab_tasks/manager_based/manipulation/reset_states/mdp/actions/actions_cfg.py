# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from . import task_space_actions


@configclass
class TransformedOperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    """Configuration for Scaled Operational Space Controller action term.

    This action term uses the OperationalSpaceController directly and applies fixed scaling
    to the input actions. The scaling values are applied per DOF (x, y, z, rx, ry, rz).
    """

    class_type: type[ActionTerm] = task_space_actions.TransformedOperationalSpaceControllerAction

    action_root_offset: OperationalSpaceControllerActionCfg.OffsetCfg | None = None
    """Offset for the action root frame."""

    scale_xyz_axisangle: tuple[float, float, float, float, float, float] = MISSING
    """Fixed scaling values for [x, y, z, rx, ry, rz] where rotation is in axis-angle representation."""

    input_clip: tuple[float, float] | None = None
    """Input clip values for the action."""
