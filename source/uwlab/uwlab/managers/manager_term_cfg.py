# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.managers.manager_term_cfg import ManagerTermBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import NoiseCfg


@configclass
class DataTermCfg(ManagerTermBaseCfg):
    """Configuration for an observation term."""

    func: Callable[..., dict] = MISSING
    """The name of the function to be called.

    This function should take the environment object and any other parameters
    as input and return the observation signal as torch float tensors of
    shape (num_envs, obs_term_dim).
    """

    noise: NoiseCfg | None = None
    """The noise to add to the observation. Defaults to None, in which case no noise is added."""

    clip: tuple[float, float] | None = None
    """The clipping range for the observation after adding noise. Defaults to None,
    in which case no clipping is applied."""

    scale: float | None = None
    """The scale to apply to the observation after clipping. Defaults to None,
    in which case no scaling is applied (same as setting scale to :obj:`1`)."""

    history_length: int = 1


@configclass
class DataGroupCfg:
    """Configuration for an observation group."""

    concatenate_terms: bool = True
    """Whether to concatenate the observation terms in the group. Defaults to True.

    If true, the observation terms in the group are concatenated along the last dimension.
    Otherwise, they are kept separate and returned as a dictionary.
    """

    enable_corruption: bool = False
    """Whether to enable corruption for the observation group. Defaults to False.

    If true, the observation terms in the group are corrupted by adding noise (if specified).
    Otherwise, no corruption is applied.
    """
