# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from io import BytesIO
from typing import TYPE_CHECKING

import requests
from isaaclab.envs.mdp.actions import JointPositionAction

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from . import actions_cfg


class PCAJointPositionAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: actions_cfg.PCAJointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.PCAJointPositionActionCfg, env: ManagerBasedEnv):
        # Download the file
        response = requests.get(cfg.eigenspace_path)
        response.raise_for_status()  # Ensure the download was successful

        # Load the numpy data from the downloaded file
        np_data = np.load(BytesIO(response.content))
        # load the the eigen_vector from the file
        self.eigenspace = torch.from_numpy(np_data).to(torch.float32)

        # initialize the action term
        super().__init__(cfg, env)

        self.eigenspace = self.eigenspace.to(self.device)
        self.joint_range = self.cfg.joint_range

    def process_actions(self, actions: torch.Tensor):
        # set position targets
        self._raw_actions[:] = actions
        self._processed_actions = self._raw_actions @ self.eigenspace * self._scale + self._offset
        self._processed_actions = torch.clamp(self.processed_actions, min=self.joint_range[0], max=self.joint_range[1])

    """
    Properties.
    """

    @property
    def action_dim(self) -> int:
        return self.eigenspace.shape[0]
