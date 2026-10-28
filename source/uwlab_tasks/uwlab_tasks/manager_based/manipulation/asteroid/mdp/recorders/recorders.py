# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.managers.recorder_manager import RecorderTerm


class PreStepExpertMaskRecorder(RecorderTerm):
    """Records, per step, whether the expert (1) or the exploration policy (0) produced the action.

    The mask is pushed in from the data-collection script (see
    ``scripts/ASTEROID/collect_demos_asteroid.py``) via :meth:`set_mask` before each step.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._expert_mask = torch.ones((env.num_envs, 1), device=env.device)
        self._exploration_horizon = None

    def set_mask(self, expert_mask: torch.Tensor):
        """Set the expert mask data externally."""
        self._expert_mask = expert_mask

    def set_exploration_horizon(self, exploration_horizon: torch.Tensor):
        """Set the exploration horizon data externally."""
        self._exploration_horizon = exploration_horizon

    def record_pre_step(self) -> tuple[str, torch.Tensor]:
        """Record the expert mask before each step."""
        return "expert_mask", self._expert_mask.clone()
