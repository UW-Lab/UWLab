# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action configs for the ASTEROID UR5e + Robotiq 2F-85 tasks.

The default 6-DOF + gripper actions are OmniReset's (re-exported here); this module adds
position-only (3-DOF + gripper) variants.
"""

from isaaclab.utils import configclass

from uwlab_assets.robots.ur5e_robotiq_gripper.actions import ROBOTIQ_GRIPPER_BINARY_ACTIONS

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.actions import (  # noqa: F401
    Ur5eRobotiq2f85RelativeOSCAction,
    Ur5eRobotiq2f85RelativeOSCEvalAction,
)

from ...mdp.actions.actions_cfg import RelCartesianOSCPositionActionCfg

# Position-only gains (mirrors the pre-train OSC gains; the policy no longer
# commands rotation and the wrist is free to rotate under collisions).
UR5E_ROBOTIQ_2F85_RELATIVE_OSC_POSONLY = RelCartesianOSCPositionActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.02, 0.02, 0.02, 0.02, 0.02, 0.2),
    motion_stiffness=(200.0, 200.0, 200.0, 3.0, 3.0, 3.0),
    motion_damping_ratio=(3.0, 3.0, 3.0, 1.0, 1.0, 1.0),
    torque_limit=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)

# Position-only eval / sim2real gains (end-of-curriculum values).
UR5E_ROBOTIQ_2F85_RELATIVE_OSC_EVAL_POSONLY = RelCartesianOSCPositionActionCfg(
    asset_name="robot",
    joint_names=["shoulder.*", "elbow.*", "wrist.*"],
    body_name="wrist_3_link",
    scale_xyz_axisangle=(0.01, 0.01, 0.002, 0.02, 0.02, 0.2),
    motion_stiffness=(1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0),
    motion_damping_ratio=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    torque_limit=(150.0, 150.0, 150.0, 28.0, 28.0, 28.0),
)


@configclass
class Ur5eRobotiq2f85RelativeOSCPositionAction:
    """Position-only action: 3-DOF Cartesian (x, y, z) arm + binary gripper.

    The policy cannot command wrist rotation -- only translate and open/close the gripper.
    Orientation is left uncommanded, so the wrist is free to rotate under collisions.
    Total action dim is 4 (3 arm + 1 gripper).
    """

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC_POSONLY
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS


@configclass
class Ur5eRobotiq2f85RelativeOSCEvalPositionAction:
    """Position-only action with high Kp gains (end-of-curriculum values) for eval / data-collection."""

    arm = UR5E_ROBOTIQ_2F85_RELATIVE_OSC_EVAL_POSONLY
    gripper = ROBOTIQ_GRIPPER_BINARY_ACTIONS
