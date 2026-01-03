# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg, NoiseModelCfg

from . import noise_model


@configclass
class OutlierNoiseCfg(NoiseCfg):
    """Configuration for an outlier noise term."""

    func = noise_model.outlier_noise

    probability: torch.Tensor | float = 0.1
    """The probability an outlier occurs, default to 0.1."""

    noise_cfg: NoiseCfg = MISSING
    """The noise function to use for outliers."""


@configclass
class NoiseModelGroupCfg(NoiseModelCfg):
    """Configuration for groups of different noise models across environments."""

    class_type: type = noise_model.NoiseModelGroup

    noise_cfg = None

    @configclass
    class NoiseModelMember:
        noise_cfg_dict: dict[str, NoiseCfg] = {}
        proportion: float = 1.0

    noise_model_groups: dict[str, NoiseModelMember] = {}
