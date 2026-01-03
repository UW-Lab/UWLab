# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from skrl.agents.torch.base import Agent
"""
loss extension functions return a loss value given a batch of samples.
they can be used to add additional loss terms to the agent's loss function.
"""


def expert_distillation_loss_f(
    env: ManagerBasedRLEnv,
    agent: Agent,
    context: dict[str, Any],
    criterion: torch.nn.Module,
    student_action_key: str = "actions",
    expert_action_key: str = "expert_actions",
) -> Callable[[dict[str, torch.Tensor]], float]:
    """
    Creates a distillation loss function to align student actions with expert actions.

    Parameters:
    ----------
    criterion : torch.nn.Module
        Loss function (e.g., `torch.nn.MSELoss`) to measure the difference between expert and student actions.
    student_action_key : str, optional
        Key to access student actions in the batch dictionary, default is "actions".
    expert_action_key : str, optional
        Key to access expert actions in the batch dictionary, default is "expert_actions".

    Returns:
    -------
    Callable[[dict], torch.Tensor]
        A function `calc_distillation_loss` that computes the distillation loss from a batch dictionary
        containing student and expert actions.

    Usage:
    ------
    ```
    distillation_loss_fn = expert_distillation_loss_f(criterion)
    loss = distillation_loss_fn(batch)
    ```
    """

    def calc_distillation_loss(batch):
        expert_action, stu_actions = batch[expert_action_key], batch[student_action_key]
        distillation_loss = criterion()(stu_actions, expert_action)
        return distillation_loss.item()

    return calc_distillation_loss
