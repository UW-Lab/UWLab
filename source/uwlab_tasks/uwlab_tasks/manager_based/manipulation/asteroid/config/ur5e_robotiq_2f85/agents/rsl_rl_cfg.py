# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from uwlab_rl.rsl_rl.rl_cfg import BehaviorCloningCfg, OffPolicyAlgorithmCfg, RslRlFancyPpoAlgorithmCfg

from uwlab_tasks.manager_based.manipulation.omnireset.config.ur5e_robotiq_2f85.agents.rsl_rl_cfg import (
    Base_PPORunnerCfg,
    my_experts_observation_func,
)


@configclass
class AsteroidPPORunnerCfg(Base_PPORunnerCfg):
    """PPO runner for the pick-only state expert (same hyperparameters as OmniReset)."""

    experiment_name = "ur5e_robotiq_2f85_asteroid_agent"


@configclass
class AsteroidDAggerRunnerCfg(AsteroidPPORunnerCfg):
    """PPO + behavior-cloning runner; the expert observes the pick-only policy observation group."""

    algorithm = RslRlFancyPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        normalize_advantage_per_mini_batch=False,
        clip_param=0.2,
        entropy_coef=0.006,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        offline_algorithm_cfg=OffPolicyAlgorithmCfg(
            behavior_cloning_cfg=BehaviorCloningCfg(
                experts_path=[""],
                experts_loader="torch.jit.load",
                experts_observation_group_cfg=(
                    "uwlab_tasks.manager_based.manipulation.asteroid.config.ur5e_robotiq_2f85.rl_state_cfg"
                    ":PickObservationsCfg.PolicyCfg"
                ),
                experts_observation_func=my_experts_observation_func,
                experts_action_group_cfg=(
                    "uwlab_tasks.manager_based.manipulation.asteroid.config.ur5e_robotiq_2f85.actions"
                    ":Ur5eRobotiq2f85RelativeOSCAction"
                ),
                cloning_loss_coeff=1.0,
                loss_decay=1.0,
            )
        ),
    )
