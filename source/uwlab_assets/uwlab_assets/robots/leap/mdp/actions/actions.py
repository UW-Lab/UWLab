# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.actions import JointAction

if TYPE_CHECKING:
    from . import actions_cfg


class LeapJointPositionActionCorrection(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.LeapJointPositionActionCorrectionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.LeapJointPositionActionCorrectionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # 0 open, 1.57 bent
        self.knuckle_idx, knuckle_names = self._asset.find_joints(["j1", "j5", "j9"], preserve_order=True)
        # -0.34 straight, +-1.04 bent
        self.tip_idx, tip_names = self._asset.find_joints(["j3", "j7", "j11"], preserve_order=True)
        # -0.4886 straight, +2.02 bent
        self.dip_idx, dip_names = self._asset.find_joints(["j2", "j6", "j10"], preserve_order=True)
        # 0 straight, +-1.04 bent
        self.mpc_idx, mpc_names = self._asset.find_joints(["j0", "j4", "j8"], preserve_order=True)

    @property
    def action_dim(self) -> int:
        return 0

    def process_actions(self, actions):
        pass

    def apply_actions(self):
        target_pos = self._asset.data.joint_pos_target.clone()
        knuckle_bending_pos = target_pos[:, self.knuckle_idx]
        target_pos[:, self.mpc_idx] -= (knuckle_bending_pos / 1.00).clamp(0, 1.00) * target_pos[:, self.mpc_idx]

        target_pos[:, self.dip_idx] = target_pos[:, self.dip_idx].clip(min=0.0)
        # print(target_pos[:, self.dip_idx])
        dip_bending_pos = target_pos[:, self.dip_idx]
        tip_max = dip_bending_pos.maximum(knuckle_bending_pos)
        tip_min = dip_bending_pos.minimum(knuckle_bending_pos) / 2
        target_pos[:, self.tip_idx] = target_pos[:, self.tip_idx].maximum(tip_min).minimum(tip_max)
        self._asset.set_joint_position_target(target_pos, self._joint_ids)
