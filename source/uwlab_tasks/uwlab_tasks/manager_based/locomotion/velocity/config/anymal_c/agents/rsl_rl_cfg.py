# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class AnymalCRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    resume = False
    experiment_name = "anymal_c_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class AnymalCRoughPositionPPORunnerCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "anymal_c_rough_position"


@configclass
class AnymalCRoughPPORunnerPrmdIprmdBoxRghHfslpCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "PrmdIprmdBoxRghHfslp"


@configclass
class AnymalCRoughPPORunnerPrmdIprmdBoxRghCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "PrmdIprmdBoxRgh"


@configclass
class AnymalCRoughPPORunnerPrmdIprmdBoxHfslpCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "PrmdIprmdBoxHfslp"


@configclass
class AnymalCRoughPPORunnerPrmdIprmdRghHfslpCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "PrmdIprmdRghHfslp"


@configclass
class AnymalCRoughPPORunnerPrmdBoxRghHfslpCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "PrmdBoxRghHfslp"


@configclass
class AnymalCRoughPPORunnerIprmdBoxRghHfslpCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "IprmdBoxRghHfslp"


@configclass
class AnymalCRoughPPORunnerBoxRghHfslpCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "BoxRghHfslp"


@configclass
class AnymalCRoughPPORunnerPrmdRghHfslpCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "PrmdRghHfslp"


@configclass
class AnymalCRoughPPORunnerPrmdIprmdHfslpCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "PrmdIprmdHfslp"


@configclass
class AnymalCRoughPPORunnerPrmdIprmdRghCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "PrmdIprmdRgh"


@configclass
class AnymalCRoughPPORunnerPrmdIprmdCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "PrmdIprmd"


@configclass
class AnymalCRoughPPORunnerIprmdBoxCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "IprmBox"


@configclass
class AnymalCRoughPPORunnerBoxRghCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "BoxRgh"


@configclass
class AnymalCRoughPPORunnerRghHfslpCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "RghHfslp"


@configclass
class AnymalCRoughPPORunnerFlatCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Flat"


@configclass
class AnymalCRoughPPORunnerBoxCfg(AnymalCRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "Box"
