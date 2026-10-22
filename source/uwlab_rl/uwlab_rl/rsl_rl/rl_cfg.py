# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg  # noqa: F401


@configclass
class BehaviorCloningCfg:
    experts_path: list[str] = MISSING  # type: ignore
    """The path to the expert data."""

    experts_loader: callable = "torch.jit.load"
    """The function to construct the expert. Default is None, for which is loaded in the same way student is loaded."""

    experts_env_mapping_func: callable = None
    """The function to map the expert to env_ids. Default is None, for which is mapped to all env_ids"""

    experts_observation_group_cfg: str | None = None
    """The observation group of the expert which may be different from student"""

    experts_observation_func: callable = None
    """The function that returns expert observation data, default is None, same as student observation."""

    experts_action_group_cfg: str | None = None
    """The action group of the expert which may be different from student"""

    learn_std: bool = False
    """Whether to learn the standard deviation of the expert policy."""

    cloning_loss_coeff: float = MISSING  # type: ignore
    """The coefficient for the cloning loss."""

    loss_decay: float = 1.0
    """The decay for the cloning loss coefficient. default to 1, no decay."""


@configclass
class OffPolicyAlgorithmCfg:
    """Configuration for the off-policy algorithm."""

    update_frequencies: float = 1
    """The frequency to update relative to online update."""

    batch_size: int | None = None
    """The batch size for the offline algorithm update, default to None, same of online size."""

    num_learning_epochs: int | None = None
    """The number of learning epochs for the offline algorithm update."""

    behavior_cloning_cfg: BehaviorCloningCfg | None = None
    """The configuration for the offline behavior cloning(dagger)."""


# Must stay an alias, not a subclass: isaaclab_rl's `policy` -> `actor`/`critic` shim
# dispatches on `type(cfg.policy) is RslRlPpoActorCriticCfg`, so a subclass falls through
# every branch and leaves `actor` MISSING (surfacing as `KeyError: 'class_name'` in
# PPO.construct_algorithm). See isaaclab_rl/rsl_rl/utils.py.
RslRlFancyActorCriticCfg = RslRlPpoActorCriticCfg
"""Alias of :class:`RslRlPpoActorCriticCfg`; see the note above."""


@configclass
class RslRlFancyPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for the PPO algorithm."""

    behavior_cloning_cfg: BehaviorCloningCfg | None = None
    """The configuration for the online behavior cloning."""

    offline_algorithm_cfg: OffPolicyAlgorithmCfg | None = None
    """The configuration for the offline algorithms."""
