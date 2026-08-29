# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Command terms for pick-only tasks."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject

from uwlab_tasks.manager_based.manipulation.omnireset.mdp.commands import TaskDependentCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import PickTaskCommandCfg


class PickTaskCommand(TaskDependentCommand):
    """Task command for pick-only tasks (a single insertive object, no receptive object).

    Counterpart of :class:`~uwlab_tasks.manager_based.manipulation.omnireset.mdp.commands.TaskCommand`
    for scenes without a receptive object. The command itself is a zero vector (the
    policy is not goal-conditioned); the term exists to drive the task-dependent reset
    events and to log pick metrics.
    """

    cfg: PickTaskCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: PickTaskCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.insertive_asset: Articulation | RigidObject = env.scene[cfg.insertive_asset_cfg.name]
        self._success_expr = cfg.success

        self.metrics["average_object_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_object_height"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["end_of_episode_success_rate"] = torch.zeros(self.num_envs, device=self.device)

        self.object_height = torch.zeros(self.num_envs, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 3, device=self.device)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs end of episode data
        reset_env = self._env.episode_length_buf == 0
        self.metrics["end_of_episode_object_height"][reset_env] = self.object_height[reset_env]
        self.metrics["end_of_episode_success_rate"][reset_env] = self.success[reset_env].float()

        # logs current data
        self.object_height[:] = self.insertive_asset.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        if self._success_expr is not None:
            self.success[:] = eval(self._success_expr, {"env": self._env})
        self.metrics["average_object_height"][:] = self.object_height

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

    def _update_command(self):
        super()._update_command()

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
