# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.utils.noise import NoiseModel

if TYPE_CHECKING:
    from . import noise_cfg


def outlier_noise(data: torch.Tensor, cfg: noise_cfg.OutlierNoiseCfg) -> torch.Tensor:
    """Applies a gaussian noise to a given data set.

    Args:
        data: The unmodified data set to apply noise to.
        cfg: The configuration parameters for gaussian noise.

    Returns:
        The data modified by the noise parameters provided.
    """

    # fix tensor device for mean on first call and update config parameters
    if isinstance(cfg.probability, torch.Tensor):
        cfg.probability = cfg.probability.to(data.device)

    # generate a mask for the outliers
    mask = torch.rand_like(data) < cfg.probability

    data[mask] = cfg.noise_cfg.func(data[mask], cfg.noise_cfg)

    return data


class NoiseModelGroup(NoiseModel):
    def __init__(self, noise_model_cfg: noise_cfg.NoiseModelGroupCfg, num_envs: int, device: str):
        # initialize parent class
        super().__init__(noise_model_cfg, num_envs, device)
        # store noise groups
        self.proportions = torch.tensor(
            [noise_group.proportion for noise_group in noise_model_cfg.noise_model_groups.values()], device=device
        )
        self.proportions /= torch.sum(self.proportions)
        self.noise_group_assignment = torch.zeros(num_envs, device=device, dtype=torch.int64)

        self.noise_model_list: list[list[noise_cfg.NoiseCfg]] = []

        for noise_group in noise_model_cfg.noise_model_groups.values():
            noise_models = list(noise_group.noise_cfg_dict.values())
            self.noise_model_list.append(noise_models)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Reset the noise model.

        This method resets resample the corresponding model group for the env_ids base on the proportion.

        Args:
            env_ids: The environment ids to reset the noise model for. Defaults to None,
                in which case all environments are considered.
        """
        # resolve the environment ids
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)

        self.noise_group_assignment[env_ids] = torch.multinomial(self.proportions, len(env_ids), replacement=True)

    def apply(self, data: torch.Tensor) -> torch.Tensor:
        """Apply bias noise to the data.

        Args:
            data: The data to apply the noise to. Shape is (num_envs, ...).

        Returns:
            The data with the noise applied. Shape is the same as the input data.
        """
        for i in range(len(self.noise_model_list)):
            noise_models = self.noise_model_list[i]
            noise_group = self.noise_group_assignment == i
            for noise_model in noise_models:
                data[noise_group] = noise_model.func(data[noise_group], noise_model)

        return data
