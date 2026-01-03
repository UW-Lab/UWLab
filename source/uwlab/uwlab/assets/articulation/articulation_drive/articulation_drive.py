# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import ArticulationDriveCfg

    # TODO: in next release remove this and should be completely


class ArticulationDrive:
    def __init__(self, cfg: ArticulationDriveCfg):
        raise NotImplementedError

    """
    Below method should be implemented by the child class
    """

    @property
    def ordered_joint_names(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def read_dof_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def write_dof_targets(self, pos_target: torch.Tensor, vel_target: torch.Tensor, eff_target: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def set_dof_stiffnesses(self, stiffnesses: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_dof_armatures(self, armatures: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_dof_frictions(self, frictions: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_dof_dampings(self, dampings: torch.Tensor) -> None:
        raise NotImplementedError

    @abstractmethod
    def set_dof_limits(self, limits: torch.Tensor) -> None:
        raise NotImplementedError
