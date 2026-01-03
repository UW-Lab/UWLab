# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction

from . import actions_cfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class TransformedOperationalSpaceControllerAction(OperationalSpaceControllerAction):
    """Scaled Operational Space Controller action term.

    This action term inherits from OperationalSpaceControllerAction and applies fixed scaling
    to the input actions. The scaling values are applied per DOF (x, y, z, rx, ry, rz) where
    rotation is in axis-angle representation.

    The workflow is:
    1. Receive 6-DOF Cartesian commands [x, y, z, rx, ry, rz] (rotation in axis-angle)
    2. Apply fixed scaling per DOF
    3. Use parent OperationalSpaceControllerAction to handle the rest
    """

    cfg: actions_cfg.TransformedOperationalSpaceControllerActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.TransformedOperationalSpaceControllerActionCfg, env: ManagerBasedEnv):
        # Initialize the parent OSC action
        super().__init__(cfg, env)

        self._scale = torch.tensor(cfg.scale_xyz_axisangle, device=self.device)
        if cfg.input_clip is not None:
            self._input_clip = torch.tensor(cfg.input_clip, device=self.device)
        else:
            self._input_clip = None
        if self.cfg.action_root_offset is not None:
            self._action_root_offset_pos = torch.tensor(cfg.action_root_offset.pos, device=self.device).repeat(
                self.num_envs, 1
            )
            self._action_root_offset_quat = torch.tensor(cfg.action_root_offset.rot, device=self.device).repeat(
                self.num_envs, 1
            )
        else:
            self._action_root_offset_pos = None
            self._action_root_offset_quat = None

        self._transformed_actions = torch.zeros_like(self.raw_actions)

    def process_actions(self, actions: torch.Tensor):
        """Process actions by applying fixed scaling per DOF and coordinate frame transformation, then call parent method."""
        # Step 1: Apply scaling
        scaled_actions_offset_coords = actions * self._scale
        if self._input_clip is not None:
            scaled_actions_offset_coords = torch.clamp(
                scaled_actions_offset_coords, min=self._input_clip[0], max=self._input_clip[1]
            )

        self._transformed_actions[:] = scaled_actions_offset_coords

        if self._action_root_offset_pos is not None and self._action_root_offset_quat is not None:
            # Step 2: Transform coordinate frame from offset-robot-base to standard-robot-base
            # Extract position and rotation deltas
            delta_pos_offset = scaled_actions_offset_coords[:, :3]  # [x, y, z]
            delta_rot_offset = scaled_actions_offset_coords[:, 3:6]  # [rx, ry, rz] in axis-angle

            # Get rotation matrix from offset-robot-base to standard-robot-base
            # The action_root_offset defines standard -> offset, so we need the inverse
            R_offset_to_standard = math_utils.matrix_from_quat(math_utils.quat_inv(self._action_root_offset_quat))

            # Transform position delta: rotate from offset coordinates to standard coordinates
            delta_pos_standard = torch.bmm(R_offset_to_standard, delta_pos_offset.unsqueeze(-1)).squeeze(-1)

            # Transform rotation delta (axis-angle): rotate the axis from offset coordinates to standard coordinates
            delta_rot_standard = torch.bmm(R_offset_to_standard, delta_rot_offset.unsqueeze(-1)).squeeze(-1)

            # Combine back into 6-DOF command
            scaled_actions_standard_coords = torch.cat([delta_pos_standard, delta_rot_standard], dim=-1)
        else:
            scaled_actions_standard_coords = scaled_actions_offset_coords

        # Call parent process_actions with transformed actions
        super().process_actions(scaled_actions_standard_coords)

    @property
    def transformed_actions(self) -> torch.Tensor:
        """Processed actions for operational space control."""
        return self._transformed_actions
