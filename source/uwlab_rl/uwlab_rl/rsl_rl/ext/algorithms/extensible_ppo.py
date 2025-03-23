# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from rsl_rl.modules import ActorCritic
from rsl_rl.modules.rnd import RandomNetworkDistillation
from rsl_rl.utils import string_to_callable

from ..storage.replay_storage import ReplayStorage
from ..storage.rollout_storage import RolloutStorage


class PPO:
    """Proximal Policy Optimization algorithm (https://arxiv.org/abs/1707.06347)."""

    actor_critic: ActorCritic
    """The actor critic module."""

    def __init__(
        self,
        actor_critic,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        # BC parameters
        behavior_cloning_cfg: dict | None = None,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Offline configuration
        offline_algorithm_cfg: dict | None = None,
    ):
        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # Online configurations
        # BC components
        if behavior_cloning_cfg is not None:
            self.bc = behavior_cloning_cfg

            if self.bc["experts_env_mapping_func"] is not None:
                self.experts_env_id_map_fn = self.bc["expert_env_id_map_fn"]
            else:
                if len(behavior_cloning_cfg["experts_path"]) > 1:
                    raise ValueError("If you have multiple experts, you need to provide a mapping function.")
                self.experts_env_id_map_fn = lambda expert_idx: slice(None)

            if self.bc["experts_observation_func"] is not None:
                self.expert_obs_fn = self.bc["expert_obs_fn"]
            else:
                self.expert_obs_shape = None  # same as student observation shape
                self.expert_critic_obs_shape = None  # same as student critic observation shape

            loader = self.bc["experts_loader"]
            if not callable(loader):
                loader = eval(loader)
            self.experts = [loader(expert_path).to(self.device).eval() for expert_path in self.bc["experts_path"]]

            self.bc_loss_coeff = self.bc["cloning_loss_coeff"]
            self.bc_decay = self.bc["loss_decay"]
            self.learn_std = self.bc["learn_std"]

        else:
            self.bc = None
            self.experts = None
            self.experts_env_id_map_fn = None

        # RND components
        if rnd_cfg is not None:
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_cfg.get("learning_rate", 1e-3))
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                warnings.warn("Symmetry not used for learning. We will use it for logging instead.")
            # If function is a string then resolve it to a function
            if isinstance(symmetry_cfg["data_augmentation_func"], str):
                symmetry_cfg["data_augmentation_func"] = string_to_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if symmetry_cfg["use_data_augmentation"] and not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    "Data augmentation enabled but the function is not callable:"
                    f" {symmetry_cfg['data_augmentation_func']}"
                )
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None  # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Offline configuration
        if offline_algorithm_cfg is not None:
            self.offline = True
            self.offline_algorithm_cfg = offline_algorithm_cfg

            self.update_counter = 0
            self.update_frequency = self.offline_algorithm_cfg["update_frequencies"]
            self.offline_batch_size: int | None = self.offline_algorithm_cfg["batch_size"]
            self.offline_num_learning_epochs: int | None = self.offline_algorithm_cfg["num_learning_epochs"]

            # Offline BC
            if "behavior_cloning_cfg" in self.offline_algorithm_cfg:
                self.offline_bc = self.offline_algorithm_cfg["behavior_cloning_cfg"]

                if self.offline_bc["experts_env_mapping_func"] is not None:
                    self.offline_experts_env_id_map_fn = self.offline_bc["expert_env_id_map_fn"]
                else:
                    if len(self.offline_bc["experts_path"]) > 1:
                        raise ValueError("If you have multiple experts, you need to provide a mapping function.")
                    self.offline_experts_env_id_map_fn = lambda expert_idx: slice(None)

                if self.offline_bc["experts_observation_func"] is not None:
                    import importlib

                    mod, attr_name = self.offline_bc["experts_observation_func"].split(":")
                    func = getattr(importlib.import_module(mod), attr_name)
                    self.offline_expert_obs_fn = func
                    self.offline_expert_obs_shape = self.offline_expert_obs_fn(self.offline_bc["_env"]).shape[1]
                else:
                    self.offline_expert_obs_fn = None
                    self.offline_expert_obs_shape = None  # same as student observation shape

                loader = self.offline_bc["experts_loader"]
                if not callable(loader):
                    loader = eval(loader)
                self.offline_experts = [
                    loader(expert_path).to(self.device).eval() for expert_path in self.offline_bc["experts_path"]
                ]

                self.offline_bc_loss_coeff = self.offline_bc["cloning_loss_coeff"]
                self.offline_bc_decay = self.offline_bc["loss_decay"]
                self.offline_learn_std = self.offline_bc["learn_std"]
            else:
                self.offline_bc = False
        else:
            self.offline = False
            self.offline_bc = False  # needed the field to exist so can be evaluated with out hasattr(self, offline_bc)

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        # create memory for RND as well :)
        if self.rnd:
            rnd_state_shape = [self.rnd.num_states]
        else:
            rnd_state_shape = None

        expert_mean_action_shape, expert_std_action_shape = None, None
        if self.bc:
            expert_mean_action_shape = action_shape
            expert_std_action_shape = action_shape if self.learn_std else None
        if self.offline_bc:
            expert_mean_action_shape = action_shape
            expert_std_action_shape = action_shape if expert_std_action_shape or self.offline_learn_std else None

        # create rollout storage
        self.storage = RolloutStorage(
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            action_shape,
            rnd_state_shape,
            expert_mean_action_shape,
            expert_std_action_shape,
            self.device,
        )
        # create replay storage
        if self.offline:
            if self.offline_bc:
                expert_mean_action_shape = action_shape
                expert_std_action_shape = action_shape if self.offline_learn_std else None
            else:
                expert_mean_action_shape, expert_std_action_shape = None, None
            self.replay_storage = ReplayStorage(
                num_envs,
                num_transitions_per_env * 20,
                actor_obs_shape,
                critic_obs_shape,
                action_shape,
                rnd_state_shape,
                expert_mean_action_shape,
                expert_std_action_shape,
                self.device,
            )

    def test_mode(self):
        self.actor_critic.test()

    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        # BC component
        if self.bc:
            for i in range(len(self.experts)):
                idx_mask = self.experts_env_id_map_fn(i)
                self.transition.expert_action_mean = self.experts[i](obs[idx_mask])
                if self.learn_std:
                    self.transition.expert_action_sigma = self.experts[i].get_actions_log_prob(
                        self.transition.expert_action_mean
                    )
        if self.offline_bc:
            for i in range(len(self.offline_experts)):
                idx_mask = self.offline_experts_env_id_map_fn(i)
                expert_obs = obs
                if self.offline_expert_obs_fn:
                    expert_obs = self.offline_expert_obs_fn(self.offline_bc["_env"])
                self.transition.expert_action_mean = self.offline_experts[i](expert_obs[idx_mask])
                if self.offline_learn_std:
                    self.transition.expert_action_sigma = self.offline_experts[i].get_actions_log_prob(
                        self.transition.expert_action_mean
                    )

        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        # Record the rewards and dones
        # Note: we clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Obtain curiosity gates / observations from infos
            rnd_state = infos["observations"]["rnd_state"]
            # Compute the intrinsic rewards
            # note: rnd_state is the gated_state after normalization if normalization is used
            self.intrinsic_rewards, rnd_state = self.rnd.get_intrinsic_reward(rnd_state)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards
            # Record the curiosity gates
            self.transition.rnd_state = rnd_state.clone()

        # Bootstrapping on time outs
        if "time_outs" in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs):
        # compute value for the last step
        last_values = self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def transfer_rollout_to_replay(self, fields: list[str] = ["observations", "privileged_observations"]):
        num_steps = self.storage.step  # number of valid transitions in rollout
        if num_steps == 0:
            return

        # Helper to perform vectorized copy into a circular buffer.
        def copy_to_replay(replay_field: torch.Tensor, rollout_field: torch.Tensor):
            # Detach and clone the slice from rollout storage.
            # shape: [num_steps, num_envs, ...]
            start_idx = self.replay_storage.step
            end_idx = start_idx + num_steps
            if end_idx <= self.replay_storage.capacity:
                # No wrap-around needed.
                replay_field[start_idx:end_idx] = rollout_field[:num_steps].detach().clone()
            else:
                # Wrap-around: split the batch copy into two parts.
                part1 = self.replay_storage.capacity - start_idx
                replay_field[start_idx:] = rollout_field[:part1].detach().clone()
                replay_field[: end_idx % self.replay_storage.capacity] = rollout_field[part1:].detach().clone()

        # Iterate over the specified fields and transfer them if available.
        for field in fields:
            copy_to_replay(getattr(self.replay_storage, field), getattr(self.storage, field))

        # Update circular buffer pointers in replay storage.
        self.replay_storage.step = (self.replay_storage.step + num_steps) % self.replay_storage.capacity
        self.replay_storage.size = min(self.replay_storage.size + num_steps, self.replay_storage.capacity)

    def update(self):  # noqa: C901
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # -- BC loss
        if self.bc:
            mean_bc_loss = 0
        else:
            mean_bc_loss = None
        # -- RND loss
        if self.rnd:
            mean_rnd_loss = 0
        else:
            mean_rnd_loss = None
        # -- Symmetry loss
        if self.symmetry:
            mean_symmetry_loss = 0
        else:
            mean_symmetry_loss = None

        # generator for mini batches
        if self.actor_critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        # iterate over batches
        for (
            obs_batch,
            critic_obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
            expert_action_mu_batch,
            expert_action_sigma_batch,
            rnd_state_batch,
        ) in generator:

            # number of augmentations per sample
            # we start with 1 and increase it if we use symmetry augmentation
            num_aug = 1
            # original batch size
            original_batch_size = obs_batch.shape[0]

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # returned shape: [batch_size * num_aug, ...]
                obs_batch, actions_batch = data_augmentation_func(
                    obs=obs_batch, actions=actions_batch, env=self.symmetry["_env"], is_critic=False
                )
                critic_obs_batch, _ = data_augmentation_func(
                    obs=critic_obs_batch, actions=None, env=self.symmetry["_env"], is_critic=True
                )
                # compute number of augmentations per sample
                num_aug = int(obs_batch.shape[0] / original_batch_size)
                # repeat the rest of the batch
                # -- actor
                old_actions_log_prob_batch = old_actions_log_prob_batch.repeat(num_aug, 1)
                # -- critic
                target_values_batch = target_values_batch.repeat(num_aug, 1)
                advantages_batch = advantages_batch.repeat(num_aug, 1)
                returns_batch = returns_batch.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: we need to do this because we updated the actor_critic with the new parameters
            # -- actor
            self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            # -- critic
            value_batch = self.actor_critic.evaluate(
                critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1]
            )
            # -- entropy
            # we only keep the entropy of the first augmentation (the original one)
            mu_batch = self.actor_critic.action_mean[:original_batch_size]
            sigma_batch = self.actor_critic.action_std[:original_batch_size]
            entropy_batch = self.actor_critic.entropy[:original_batch_size]

            # KL
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            # Symmetry loss
            if self.symmetry:
                # obtain the symmetric actions
                # if we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    obs_batch, _ = data_augmentation_func(
                        obs=obs_batch, actions=None, env=self.symmetry["_env"], is_critic=False
                    )
                    # compute number of augmentations per sample
                    num_aug = int(obs_batch.shape[0] / original_batch_size)

                # actions predicted by the actor for symmetrically-augmented observations
                mean_actions_batch = self.actor_critic.act_inference(obs_batch.detach().clone())

                # compute the symmetrically augmented actions
                # note: we are assuming the first augmentation is the original one.
                #   We do not use the action_batch from earlier since that action was sampled from the distribution.
                #   However, the symmetry loss is computed using the mean of the distribution.
                action_mean_orig = mean_actions_batch[:original_batch_size]
                _, actions_mean_symm_batch = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"], is_critic=False
                )

                # compute the loss (we skip the first augmentation as it is the original one)
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions_batch[original_batch_size:], actions_mean_symm_batch.detach()[original_batch_size:]
                )
                # add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()
            # BC loss
            if self.bc:
                mse_loss = torch.nn.MSELoss()
                mean_loss = mse_loss(mu_batch, expert_action_mu_batch)
                bc_loss = mean_loss
                if self.learn_std:
                    std_loss = mse_loss(sigma_batch, expert_action_sigma_batch)
                    bc_loss += std_loss
                self.bc_loss_coeff *= self.bc_decay
                loss = (1 - self.bc_loss_coeff) * loss + self.bc_loss_coeff * bc_loss

            # Random Network Distillation loss
            if self.rnd:
                # predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state_batch)
                target_embedding = self.rnd.target(rnd_state_batch)
                # compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding.detach())

            # Gradient step
            # -- For PPO
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # -- For RND
            if self.rnd_optimizer:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            # -- BC loss
            if mean_bc_loss is not None:
                mean_bc_loss += bc_loss.item()
            # -- RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # -- Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # -- For PPO
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        # -- For BC
        if mean_bc_loss is not None:
            mean_bc_loss /= num_updates
        # -- For RND
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        # -- For Symmetry
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        # -- Clear the storage

        if self.offline:
            fields_to_transfer = ["observations", "privileged_observations"]
            if self.offline_bc:
                fields_to_transfer.append("expert_action_mean")
                if self.offline_learn_std:
                    fields_to_transfer.append("expert_action_sigma")
            self.transfer_rollout_to_replay(fields_to_transfer)
            while self.update_counter >= 0:
                capacity_percentage, offline_bc_loss = self.update_offline()
                self.update_counter -= 1 / self.update_frequency
            self.update_counter += 1
        else:
            capacity_percentage, offline_bc_loss = None, None
        self.storage.clear()

        return (
            mean_value_loss,
            mean_surrogate_loss,
            mean_entropy,
            mean_bc_loss,
            mean_rnd_loss,
            mean_symmetry_loss,
            capacity_percentage,
            offline_bc_loss,
        )

    def update_offline(self):
        loss = 0
        storage_data_size = self.replay_storage.num_envs * self.replay_storage.size

        batch_size = (
            self.offline_batch_size
            if self.offline_batch_size
            else self.storage.num_envs * self.storage.num_transitions_per_env // self.num_mini_batches
        )
        num_mini_batches = storage_data_size // batch_size
        num_learning_epoch = (
            self.offline_num_learning_epochs if self.offline_num_learning_epochs else self.num_learning_epochs
        )

        if self.actor_critic.is_recurrent:
            generator = self.replay_storage.recurrent_mini_batch_generator(num_mini_batches, num_learning_epoch)
        else:
            generator = self.replay_storage.mini_batch_generator(num_mini_batches, num_learning_epoch)

        # -- For BC
        if self.offline_bc:
            mse_loss_fn = nn.MSELoss()
            mean_bc_loss = 0.0
        else:
            mean_bc_loss = None

        for obs_batch, _, _, _, _, _, _, _, _, _, _, expert_action_mu_batch, expert_action_sigma_batch, _ in generator:
            # -- For BC
            if self.offline_bc:
                predicted_actions = self.actor_critic.act_inference(obs_batch)
                bc_loss = mse_loss_fn(predicted_actions, expert_action_mu_batch)
                if self.offline_learn_std:
                    predicted_std = self.actor_critic.action_std[: obs_batch.shape[0]]
                    bc_loss += mse_loss_fn(predicted_std, expert_action_sigma_batch)
                self.offline_bc_loss_coeff *= self.offline_bc_decay
                loss = (1 - self.offline_bc_loss_coeff) * loss + self.offline_bc_loss_coeff * bc_loss

            self.optimizer.zero_grad()
            bc_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # --BC loss
            if mean_bc_loss is not None:
                mean_bc_loss += bc_loss.item()

        num_updates = num_mini_batches * num_learning_epoch
        # For BC
        if mean_bc_loss is not None:
            mean_bc_loss /= num_updates
        capacity_percentage = self.replay_storage.size / self.replay_storage.capacity
        return capacity_percentage, mean_bc_loss
