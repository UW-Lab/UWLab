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

from rsl_rl.utils import split_and_pad_trajectories


class ReplayStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.critic_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None

            # For Policy Gradient-like algorithms
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

            # For RNN-based policies
            self.hidden_states = None

            # For RND
            self.rnd_state = None

            # For BC (behavior cloning)
            self.expert_action_mean = None
            self.expert_action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        capacity: int,
        obs_shape=None,
        privileged_obs_shape=None,
        actions_shape=None,
        rnd_state_shape=None,
        expert_actions_shape=None,
        expert_actions_sigma_shape=None,
        device="cpu",
    ):
        # store inputs
        self.device = device
        self.capacity = capacity
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.privileged_obs_shape = privileged_obs_shape
        self.rnd_state_shape = rnd_state_shape
        self.actions_shape = actions_shape
        self.expert_action_mean_shape = expert_actions_shape
        self.expert_action_sigma_shape = expert_actions_sigma_shape

        # Core
        if obs_shape is not None:
            self.observations = torch.zeros(capacity, num_envs, *obs_shape, device=device)
        if privileged_obs_shape is not None:
            self.privileged_observations = torch.zeros(capacity, num_envs, *privileged_obs_shape, device=device)
        else:
            self.privileged_observations = None

        if actions_shape is not None:
            self.rewards = torch.zeros(capacity, num_envs, 1, device=self.device)
            self.actions = torch.zeros(capacity, num_envs, *actions_shape, device=self.device)
            self.dones = torch.zeros(capacity, num_envs, 1, device=self.device).byte()

            # For Policy-Gradient-like algorithms
            self.actions_log_prob = torch.zeros(capacity, num_envs, 1, device=self.device)
            self.values = torch.zeros(capacity, num_envs, 1, device=self.device)
            self.returns = torch.zeros(capacity, num_envs, 1, device=self.device)
            self.advantages = torch.zeros(capacity, num_envs, 1, device=self.device)
            self.mu = torch.zeros(capacity, num_envs, *actions_shape, device=self.device)
            self.sigma = torch.zeros(capacity, num_envs, *actions_shape, device=self.device)

        # For BC
        if expert_actions_shape is not None:
            self.expert_action_mean = torch.zeros(capacity, num_envs, *expert_actions_shape, device=self.device)
        if expert_actions_sigma_shape is not None:
            self.expert_action_sigma = torch.zeros(capacity, num_envs, *expert_actions_sigma_shape, device=self.device)

        # For RND
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(capacity, num_envs, *rnd_state_shape, device=self.device)

        # For RNN networks
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

        # Circular buffer pointers
        self.step = 0
        self.size = 0
        self.is_full = False

    def add_transitions(self, transition: Transition):
        # check if the transition is valid
        if self.size >= self.capacity:
            self.is_full = True
        # Core
        if self.obs_shape is not None:
            self.observations[self.step].copy_(transition.observations)
        if self.privileged_observations is not None:
            self.privileged_observations[self.step].copy_(transition.critic_observations)
        if self.actions_shape is not None:
            self.actions[self.step].copy_(transition.actions)
            self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
            self.dones[self.step].copy_(transition.dones.view(-1, 1))

            # For Gradient-like algorithms
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
            self.mu[self.step].copy_(transition.action_mean)
            self.sigma[self.step].copy_(transition.action_sigma)

        # For BC
        if transition.expert_action_mean is not None and self.expert_action_mean is not None:
            self.expert_action_mean[self.step].copy_(transition.expert_action_mean)
        if transition.expert_action_sigma is not None and self.expert_action_sigma is not None:
            self.expert_action_sigma[self.step].copy_(transition.expert_action_sigma)

        # For RND
        if self.rnd_state_shape is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)

        # For RNN networks
        self._save_hidden_states(transition.hidden_states)

        # Update circular buffer pointers
        self.step = (self.step + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _save_hidden_states(self, hidden_states):
        if hidden_states is None or hidden_states == (None, None):
            return
        # make a tuple out of GRU hidden state sto match the LSTM format
        hid_a = hidden_states[0] if isinstance(hidden_states[0], tuple) else (hidden_states[0],)
        hid_c = hidden_states[1] if isinstance(hidden_states[1], tuple) else (hidden_states[1],)

        # initialize if needed
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = [
                torch.zeros(self.observations.shape[0], *hid_a[i].shape, device=self.device) for i in range(len(hid_a))
            ]
            self.saved_hidden_states_c = [
                torch.zeros(self.observations.shape[0], *hid_c[i].shape, device=self.device) for i in range(len(hid_c))
            ]
        # copy the states
        for i in range(len(hid_a)):
            self.saved_hidden_states_a[i][self.step].copy_(hid_a[i])
            self.saved_hidden_states_c[i][self.step].copy_(hid_c[i])

    def clear(self):
        self.step = 0
        self.size = 0
        self.is_full = False

    def compute_returns(self, last_values, gamma, lam):
        advantage = 0
        for step in reversed(range(self.size)):
            # if we are at the last step, bootstrap the return value
            if step == self.size - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            self.returns[step] = advantage + self.values[step]

        # Compute and normalize the advantages
        self.advantages = self.returns[: self.size] - self.values[: self.size]
        self.advantages[: self.size] = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self):
        done = self.dones[: self.size]
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards[: self.size].mean()

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.size
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        # Core
        if self.observations is not None:
            observations = self.observations[: self.size].flatten(0, 1)
        if self.privileged_observations is not None:
            critic_observations = self.privileged_observations[: self.size].flatten(0, 1)
        else:
            critic_observations = observations

        if self.actions is not None:
            actions = self.actions[: self.size].flatten(0, 1)
            values = self.values[: self.size].flatten(0, 1)
            returns = self.returns[: self.size].flatten(0, 1)

            # For PPO
            old_actions_log_prob = self.actions_log_prob[: self.size].flatten(0, 1)
            advantages = self.advantages[: self.size].flatten(0, 1)
            old_mu = self.mu[: self.size].flatten(0, 1)
            old_sigma = self.sigma[: self.size].flatten(0, 1)
        # For BC
        if self.expert_action_mean_shape is not None:
            expert_action_mu = self.expert_action_mean[: self.size].flatten(0, 1)
        if self.expert_action_sigma_shape is not None:
            expert_action_sigma = self.expert_action_sigma[: self.size].flatten(0, 1)

        # For RND
        if self.rnd_state_shape is not None:
            rnd_state = self.rnd_state[: self.size].flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                # Create the mini-batch
                # -- Core
                obs_batch, critic_observations_batch, actions_batch = None, None, None
                if self.observations is not None:
                    obs_batch = observations[batch_idx]
                    critic_observations_batch = critic_observations[batch_idx]
                if self.actions is not None:
                    actions_batch = actions[batch_idx]

                # -- For PPO
                target_values_batch, returns_batch, advantages_batch = None, None, None
                old_actions_log_prob_batch, old_mu_batch, old_sigma_batch = None, None, None
                if self.actions is not None:
                    target_values_batch = values[batch_idx]
                    returns_batch = returns[batch_idx]
                    old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                    advantages_batch = advantages[batch_idx]
                    old_mu_batch = old_mu[batch_idx]
                    old_sigma_batch = old_sigma[batch_idx]
                # -- For BC
                expert_action_mu_batch, expert_action_sigma_batch = None, None
                if self.expert_action_mean_shape is not None:
                    expert_action_mu_batch = expert_action_mu[batch_idx]
                if self.expert_action_sigma_shape is not None:
                    expert_action_sigma_batch = expert_action_sigma[batch_idx]

                # -- For RND
                rnd_state_batch = None
                if self.rnd_state_shape is not None:
                    rnd_state_batch = rnd_state[batch_idx]

                # Yield the mini-batch
                yield obs_batch, critic_observations_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    None,
                    None,
                ), None, expert_action_mu_batch, expert_action_sigma_batch, rnd_state_batch

    # for RNNs only
    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        padded_obs_trajectories, trajectory_masks = split_and_pad_trajectories(self.observations, self.dones)
        if self.privileged_observations is not None:
            padded_critic_obs_trajectories, _ = split_and_pad_trajectories(self.privileged_observations, self.dones)
        else:
            padded_critic_obs_trajectories = padded_obs_trajectories

        if self.rnd_state_shape is not None:
            padded_rnd_state_trajectories, _ = split_and_pad_trajectories(self.rnd_state, self.dones)
        else:
            padded_rnd_state_trajectories = None

        mini_batch_size = self.num_envs // num_mini_batches
        for ep in range(num_epochs):
            first_traj = 0
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size

                dones = self.dones.squeeze(-1)
                last_was_done = torch.zeros_like(dones, dtype=torch.bool)
                last_was_done[1:] = dones[:-1]
                last_was_done[0] = True
                trajectories_batch_size = torch.sum(last_was_done[:, start:stop])
                last_traj = first_traj + trajectories_batch_size

                masks_batch = trajectory_masks[:, first_traj:last_traj]
                obs_batch = padded_obs_trajectories[:, first_traj:last_traj]
                critic_obs_batch = padded_critic_obs_trajectories[:, first_traj:last_traj]

                # For BC
                if self.expert_action_mean_shape is not None:
                    expert_action_mu_batch = self.expert_action_mean[:, start:stop]
                else:
                    expert_action_mu_batch = None
                if self.expert_action_sigma_shape is not None:
                    expert_action_sigma_batch = self.expert_action_sigma[:, start:stop]
                else:
                    expert_action_sigma_batch = None

                if padded_rnd_state_trajectories is not None:
                    rnd_state_batch = padded_rnd_state_trajectories[:, first_traj:last_traj]
                else:
                    rnd_state_batch = None
                actions_batch = self.actions[:, start:stop]
                old_mu_batch = self.mu[:, start:stop]
                old_sigma_batch = self.sigma[:, start:stop]
                returns_batch = self.returns[:, start:stop]
                advantages_batch = self.advantages[:, start:stop]
                values_batch = self.values[:, start:stop]
                old_actions_log_prob_batch = self.actions_log_prob[:, start:stop]

                # reshape to [num_envs, time, num layers, hidden dim] (original shape: [time, num_layers, num_envs, hidden_dim])
                # then take only time steps after dones (flattens num envs and time dimensions),
                # take a batch of trajectories and finally reshape back to [num_layers, batch, hidden_dim]
                last_was_done = last_was_done.permute(1, 0)
                hid_a_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_a
                ]
                hid_c_batch = [
                    saved_hidden_states.permute(2, 0, 1, 3)[last_was_done][first_traj:last_traj]
                    .transpose(1, 0)
                    .contiguous()
                    for saved_hidden_states in self.saved_hidden_states_c
                ]
                # remove the tuple for GRU
                hid_a_batch = hid_a_batch[0] if len(hid_a_batch) == 1 else hid_a_batch
                hid_c_batch = hid_c_batch[0] if len(hid_c_batch) == 1 else hid_c_batch

                yield obs_batch, critic_obs_batch, actions_batch, values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, (
                    hid_a_batch,
                    hid_c_batch,
                ), masks_batch, expert_action_mu_batch, expert_action_sigma_batch, rnd_state_batch

                first_traj = last_traj
