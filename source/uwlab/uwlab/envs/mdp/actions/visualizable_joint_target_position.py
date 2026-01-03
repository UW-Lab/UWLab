# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets.articulation import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm

if TYPE_CHECKING:
    from . import actions_cfg


class VisualizableJointTargetPosition(ActionTerm):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.VisualizableJointTargetPositionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return torch.tensor([])

    @property
    def processed_actions(self) -> torch.Tensor:
        return torch.tensor([])

    def process_actions(self, actions):
        pass

    def apply_actions(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        import isaacsim.core.utils.prims as prim_utils
        from pxr import UsdGeom

        if debug_vis:
            if not hasattr(self, "vis_articulation"):
                if self.cfg.articulation_vis_cfg.name in self._env.scene:
                    self.vis_articulation: Articulation = self._env.scene[self.cfg.articulation_vis_cfg.name]
                    prims_paths = prim_utils.find_matching_prim_paths(self.vis_articulation.cfg.prim_path)
                    prims = [prim_utils.get_prim_at_path(prim) for prim in prims_paths]
                    for prim in prims:
                        UsdGeom.Imageable(prim).MakeVisible()
        else:
            if hasattr(self, "vis_articulation"):
                prims_paths = prim_utils.find_matching_prim_paths(self.vis_articulation.cfg.prim_path)
                prims = [prim_utils.get_prim_at_path(prim) for prim in prims_paths]
                for prim in prims:
                    UsdGeom.Imageable(prim).MakeInvisible()

    def _debug_vis_callback(self, event):
        # update the box marker
        self.vis_articulation.write_joint_state_to_sim(
            position=self._asset.data.joint_pos_target,
            velocity=torch.zeros_like(self._asset.data.joint_pos_target, device=self.device),
        )
