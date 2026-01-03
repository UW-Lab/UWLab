# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.envs.mdp import JointActionCfg, JointPositionActionCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from uwlab.envs.mdp.actions import (
    default_joint_static_action,
    pca_actions,
    task_space_actions,
    visualizable_joint_target_position,
)

##
# Task-space Actions.
##


@configclass
class MultiConstraintsDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for inverse differential kinematics action term with multi constraints.
    This class amend attr body_name from type:str to type:list[str] reflecting its capability to
    received the desired positions, poses from multiple target bodies. This will be particularly
    useful for controlling dextrous hand robot with only positions of multiple key frame positions
    and poses, and output joint positions that satisfy key frame position/pose constrains

    See :class:`DifferentialInverseKinematicsAction` for more details.
    """

    @configclass
    class OffsetCfg:
        @configclass
        class BodyOffsetCfg:
            """The offset pose from parent frame to child frame.

            On many robots, end-effector frames are fictitious frames that do not have a corresponding
            rigid body. In such cases, it is easier to define this transform w.r.t. their parent rigid body.
            For instance, for the Franka Emika arm, the end-effector is defined at an offset to the the
            "panda_hand" frame.
            """

            pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
            """Translation w.r.t. the parent frame. Defaults to (0.0, 0.0, 0.0)."""
            rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
            """Quaternion rotation ``(w, x, y, z)`` w.r.t. the parent frame. Defaults to (1.0, 0.0, 0.0, 0.0)."""

        pose: dict[str, BodyOffsetCfg] = {}

    body_name: list[str] = MISSING

    class_type: type[ActionTerm] = task_space_actions.MultiConstraintDifferentialInverseKinematicsAction

    body_offset: OffsetCfg | None = None

    task_space_boundary: list[tuple[float, float]] | None = None


@configclass
class PCAJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = pca_actions.PCAJointPositionAction

    eigenspace_path: str = MISSING

    joint_range: tuple[float, float] | dict[str, tuple[float, float]] = MISSING


@configclass
class VisualizableJointTargetPositionCfg(JointActionCfg):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    class_type: type[ActionTerm] = visualizable_joint_target_position.VisualizableJointTargetPosition

    articulation_vis_cfg: SceneEntityCfg = MISSING


@configclass
class DefaultJointPositionStaticActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = default_joint_static_action.DefaultJointPositionStaticAction

    use_default_offset: bool = True
