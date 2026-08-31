# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from uwlab_tasks.manager_based.manipulation.omnireset.mdp.actions.task_space_actions import RelCartesianOSCAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class RelCartesianOSCPositionAction(RelCartesianOSCAction):
    """Position-only variant of :class:`RelCartesianOSCAction`.

    The policy outputs a 3-DOF Cartesian delta ``[x, y, z]`` only -- it cannot
    rotate the gripper. The desired EE orientation simply tracks the *current*
    orientation each policy step, so the controller never accumulates a rotation
    error to fight: collision-induced rotations are accepted rather than resisted.

    All control machinery (analytical Jacobian, PD torques, clamping) is
    inherited unchanged from the parent; only the action interface and the
    desired-pose computation differ.
    """

    cfg: actions_cfg.RelCartesianOSCPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.RelCartesianOSCPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Override the 6-DOF action buffers with 3-DOF (x, y, z) ones.
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def action_dim(self) -> int:
        return 3

    def process_actions(self, actions: torch.Tensor):
        """Scale raw 3-DOF xyz deltas and compute the desired EE position.

        The desired orientation tracks the current EE orientation, so no rotation
        error builds up and collision-induced rotations are accepted, not fought.
        """
        self._raw_actions[:] = actions
        # ``_scale`` is ``(6,)`` by default and ``(num_envs, 6)`` once domain randomization
        # (:class:`~..events.randomize_env_cfg_unified`) has written per-env scales into it.
        scaled = actions * self._scale[..., :3]
        if self._input_clip is not None:
            scaled = torch.clamp(scaled, min=self._input_clip[0], max=self._input_clip[1])
        self._processed_actions[:] = scaled

        # Current EE pose in root (base_link) frame.
        ee_pos_b, ee_quat_b = self._get_ee_pose_root_frame()
        # Position: desired = current + delta. Orientation: track current (no command).
        self._ee_pos_des[:] = ee_pos_b + scaled
        self._ee_quat_des[:] = ee_quat_b
