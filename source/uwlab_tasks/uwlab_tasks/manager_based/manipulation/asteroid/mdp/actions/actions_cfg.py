# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from uwlab_tasks.manager_based.manipulation.omnireset.mdp.actions.actions_cfg import RelCartesianOSCActionCfg

from . import task_space_actions


@configclass
class RelCartesianOSCPositionActionCfg(RelCartesianOSCActionCfg):
    """Position-only Relative Cartesian OSC action.

    Identical to :class:`RelCartesianOSCActionCfg` except the policy controls
    only the 3-DOF Cartesian position ``[x, y, z]``; the gripper orientation is
    left uncommanded and the desired orientation tracks the current orientation,
    so collision-induced rotations are accepted rather than resisted. The full
    6-tuple gain / scale / torque fields are retained (the rotation gains only
    provide light damping within a policy step).
    """

    class_type: type[ActionTerm] = task_space_actions.RelCartesianOSCPositionAction
