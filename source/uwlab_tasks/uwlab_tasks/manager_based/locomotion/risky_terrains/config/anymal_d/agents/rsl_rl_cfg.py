# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg

from ..augment import aug_func


def my_experts_observation_func(env):
    return env.unwrapped.obs_buf["expert_obs"]


@configclass
class RiskyTerrainsAnymalDPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 48
    max_iterations = 8000
    save_interval = 400
    resume = False
    experiment_name = "anymal_d_risky_terrains"
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
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True, use_mirror_loss=False, data_augmentation_func=aug_func
        ),
        # offline_algorithm_cfg=OffPolicyAlgorithmCfg(
        #     behavior_cloning_cfg=BehaviorCloningCfg(
        #         experts_path=["logs/rsl_rl/UW-Stepping-Stone-Anymal-D-v0/2025-02-14_23-05-48/exported/policy.pt"],
        #         experts_observation_group_cfg="uwlab_tasks.tasks.locomotion.risky_terrains.stepping_stones_env:ObservationsCfg.PolicyCfg",
        #         experts_observation_func=my_experts_observation_func,
        #         cloning_loss_coeff=1.0,
        #         loss_decay=1.0
        #     )
        # )
    )
