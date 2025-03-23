# base_config.py
# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING, field
from typing import Dict, List, Literal, Optional, Union

from isaaclab.utils import configclass

from .extensions import SupplementaryTrainingCfg


@configclass
class ModelPolicyCfg:
    """Base configuration for the policy model."""

    policy_class: Literal["GaussianMixin", "MultivariateGaussianMixin"] = field(default="GaussianMixin")
    clip_actions: bool = MISSING
    clip_log_std: bool = MISSING
    initial_log_std: float = MISSING
    min_log_std: float = MISSING
    max_log_std: float = MISSING
    input_shape: str = MISSING
    hiddens: List[int] = MISSING
    hidden_activation: List[str] = MISSING
    output_shape: str = MISSING
    output_activation: str = MISSING
    output_scale: float = MISSING


@configclass
class ModelValueCfg:
    """Base configuration for the value model."""

    value_class: str = MISSING
    clip_actions: bool = MISSING
    input_shape: str = MISSING
    hiddens: List[int] = MISSING
    hidden_activation: List[str] = MISSING
    output_shape: str = MISSING
    output_activation: str = MISSING
    output_scale: float = MISSING


@configclass
class ModelsCfg:
    """Base configuration for the model instantiators."""

    separate: bool = MISSING
    policy: ModelPolicyCfg = MISSING
    value: ModelValueCfg = MISSING


@configclass
class ExperimentCfg:
    """Base configuration for experiment logging and checkpoints."""

    directory: str = MISSING
    experiment_name: str = MISSING
    write_interval: int = MISSING
    checkpoint_interval: int = MISSING
    wandb: bool = MISSING
    wandb_kwargs: Dict[str, str] = field(default_factory=dict)


@configclass
class PPOAgentCfg:
    """Base configuration for the PPO agent."""

    agent_class: str = MISSING
    rollouts: int = MISSING
    learning_epochs: int = MISSING
    mini_batches: int = MISSING
    discount_factor: float = MISSING
    lambda_: float = MISSING
    learning_rate: float = MISSING
    learning_rate_scheduler: str = MISSING
    learning_rate_scheduler_kwargs: Dict[str, Union[float, str]] = field(default_factory=dict)
    state_preprocessor: str = MISSING
    state_preprocessor_kwargs: Optional[Dict] = None
    value_preprocessor: str = MISSING
    value_preprocessor_kwargs: Optional[Dict] = None
    random_timesteps: int = MISSING
    learning_starts: int = MISSING
    grad_norm_clip: float = MISSING
    ratio_clip: float = MISSING
    value_clip: float = MISSING
    clip_predicted_values: bool = MISSING
    entropy_loss_scale: float = MISSING
    value_loss_scale: float = MISSING
    kl_threshold: float = MISSING
    rewards_shaper_scale: float = MISSING
    experiment: ExperimentCfg = MISSING


@configclass
class TrainerCfg:
    """Base configuration for the sequential trainer."""

    timesteps: int = MISSING
    environment_info: str = MISSING
    close_environment_at_exit: bool = True


@configclass
class ExtensionCfg:
    supplementary_training_cfg: SupplementaryTrainingCfg = MISSING


@configclass
class SKRLConfig:
    """Base configuration for the RL agent and training setup."""

    seed: int = MISSING
    models: ModelsCfg = MISSING
    agent: PPOAgentCfg = MISSING
    trainer: TrainerCfg = MISSING
    extension: Optional[ExtensionCfg] = None

    def to_skrl_dict(self):
        dict = self.to_dict()
        dict["agent"]["class"] = dict["agent"].pop("agent_class", None)
        dict["models"]["policy"]["class"] = dict["models"]["policy"].pop("policy_class", None)
        dict["models"]["value"]["class"] = dict["models"]["value"].pop("value_class", None)
        return dict
