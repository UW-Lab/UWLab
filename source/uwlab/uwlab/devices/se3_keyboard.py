# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.devices import Se3Keyboard
from isaaclab.utils.math import matrix_from_quat

if TYPE_CHECKING:
    from uwlab.devices import KeyboardCfg


class Se3Keyboard(Se3Keyboard):
    """A keyboard controller for sending SE(3) commands as absolute poses and binary command (open/close).

    This class is designed to provide a keyboard controller for a robotic arm with a gripper.
    It uses the Omniverse keyboard interface to listen to keyboard events and map them to robot's
    task-space commands. Different from Isaac Lab Se3Keyboard that adds delta command to current robot pose,
    this implementation controls a target pose which robot actively tracks. this error corrective property is
    advantageous to use when robot is prone to drift or under actuated

    The command comprises of two parts:

    * absolute pose: a 6D vector of (x, y, z, roll, pitch, yaw) in meters and radians.
    * gripper: a binary command to open or close the gripper.

    Key bindings:
        ============================== ================= =================
        Description                    Key (+ve axis)    Key (-ve axis)
        ============================== ================= =================
        Toggle gripper (open/close)    K
        Move along x-axis              W                 S
        Move along y-axis              A                 D
        Move along z-axis              Q                 E
        Rotate along x-axis            Z                 X
        Rotate along y-axis            T                 G
        Rotate along z-axis            C                 V
        ============================== ================= =================

    .. seealso::

        The official documentation for the keyboard interface: `Carb Keyboard Interface <https://docs.omniverse.nvidia.com/dev-guide/latest/programmer_ref/input-devices/keyboard.html>`__.

    """

    def __init__(self, cfg: KeyboardCfg, device="cuda:0"):
        """Initialize the keyboard layer.

        Args:
            pos_sensitivity:    Magnitude of input position command scaling. Defaults to 0.05.
            rot_sensitivity:    Magnitude of scale input rotation commands scaling. Defaults to 0.5.
        """
        super().__init__(cfg.pos_sensitivity, cfg.rot_sensitivity)
        self.device = device
        self.enable_gripper_command = cfg.enable_gripper_command
        self.init_pose = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], device=self.device)
        self.abs_pose = torch.zeros((1, 6), device=self.device)

    """
    Operations
    """

    def reset(self):
        super().reset()
        self.abs_pose = self.init_pose.clone()

    def advance(self):
        """Provides the result from internal target command modified by keyboard event.

        Returns:
            A tuple containing the absolute pose command and gripper commands.
        """
        delta_pose, gripper_command = super().advance()
        delta_pose = delta_pose.astype("float32")
        # convert to torch
        delta_pose = torch.tensor(delta_pose, device=self.device).view(1, -1)
        self.abs_pose[:, :3] += delta_pose[:, :3]
        self.abs_pose[:, 3:] += delta_pose[:, 3:]

        if self.enable_gripper_command:
            if gripper_command:
                gripper_command = torch.tensor([[1.0]], device=self.device)
            else:
                gripper_command = torch.tensor([[-1.0]], device=self.device)
            return self.abs_pose.clone(), gripper_command
        else:
            return self.abs_pose.clone()


def apply_local_translation(current_position, local_translation, orientation_quaternion):
    # Assuming matrix_from_quat correctly handles batch inputs and outputs a batch of rotation matrices
    rotation_matrix = matrix_from_quat(orientation_quaternion)  # Expected shape (n, 3, 3)

    # Ensure local_translation is correctly shaped for batch matrix multiplication
    local_translation = local_translation.unsqueeze(-1)  # Shape becomes (n, 3, 1) for matmul

    local_translation[:, [1, 2]] = -local_translation[:, [2, 1]]
    # Rotate the local translation vector to align with the global frame
    global_translation = torch.matmul(rotation_matrix, local_translation).squeeze(-1)  # Back to shape (n, 3)

    # Apply the translated vector to the object's current position
    new_position = current_position + global_translation

    return new_position
