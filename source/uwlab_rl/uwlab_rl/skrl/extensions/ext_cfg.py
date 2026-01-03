# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Callable
from dataclasses import MISSING
from typing import Any

from isaaclab.utils import configclass


@configclass
class ContextInitializerCfg:
    """context initializer for the supplementary training."""

    context_identifier: str = MISSING

    context_initializer: Callable[..., dict[str, Any]] = MISSING

    context_params: dict[str, Any] = MISSING


@configclass
class SupplementaryLossesCfg:
    """additional loss term for the training."""

    loss_term: str = MISSING

    loss_f_creator: Callable[..., Callable[[dict[str, torch.Tensor]], float]] = MISSING

    loss_params: dict[str, Any] = MISSING


@configclass
class SupplementarySampleTermsCfg:
    """additional sample term for the training."""

    sample_term: str = MISSING

    sample_size_f_creator: Callable[..., Callable[[], int]] = MISSING

    sample_size_params: dict[str, Any] = MISSING

    sample_f_creator: Callable[..., Callable[[dict[str, torch.Tensor]], torch.Tensor]] = MISSING

    sample_params: dict[str, Any] = MISSING


@configclass
class SupplementaryTrainingCfg:
    context_manager: Callable = MISSING

    context_initializers: list[ContextInitializerCfg] = MISSING

    supplementary_losses: list[SupplementaryLossesCfg] = MISSING

    supplementary_sample_terms: list[SupplementarySampleTermsCfg] = MISSING
