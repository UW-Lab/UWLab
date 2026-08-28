# Copyright (c) 2026, Null-space preference critic project.
# SPDX-License-Identifier: BSD-3-Clause

"""Rollout storage carrying two reward/value/return/advantage streams.

The two streams get **separately normalised advantages**. That is the detail that makes the
reward model's arbitrary output scale cancel, so β controls direction budget only and does not
have to be re-tuned per task or per reward-model retrain.

Upstream normalises once over the whole (num_steps, num_envs) batch rather than per minibatch
(``normalize_advantage_per_mini_batch=False`` in the OmniReset agent config), so each stream is
normalised over its own full batch here.
"""

from __future__ import annotations

import torch
from collections.abc import Generator
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage


class DualRolloutStorage(RolloutStorage):
    """RolloutStorage plus a parallel preference stream."""

    class Transition(RolloutStorage.Transition):
        def __init__(self) -> None:
            super().__init__()
            self.rewards_pref: torch.Tensor | None = None
            self.values_pref: torch.Tensor | None = None

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        super().__init__(training_type, num_envs, num_transitions_per_env, obs, actions_shape, device)

        if training_type != "rl":
            raise ValueError("DualRolloutStorage only supports training_type='rl'.")

        z = lambda: torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)  # noqa: E731
        self.rewards_pref = z()
        self.values_pref = z()
        self.returns_pref = z()
        self.advantages_pref = z()

    def add_transitions(self, transition: Transition) -> None:
        # NOTE: the base implementation increments self.step, so capture it first.
        step = self.step
        super().add_transitions(transition)
        self.rewards_pref[step].copy_(transition.rewards_pref.view(-1, 1))
        self.values_pref[step].copy_(transition.values_pref)

    # -- GAE ---------------------------------------------------------------------------------

    @staticmethod
    def _gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        num_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Standard GAE. Returns (returns, advantages), unnormalised."""
        returns = torch.zeros_like(values)
        advantage = 0
        for step in reversed(range(num_steps)):
            next_values = last_values if step == num_steps - 1 else values[step + 1]
            next_is_not_terminal = 1.0 - dones[step].float()
            delta = rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns[step] = advantage + values[step]
        return returns, returns - values

    def compute_returns_dual(
        self,
        last_values: torch.Tensor,
        last_values_pref: torch.Tensor,
        gamma: float,
        lam: float,
        gamma_pref: float,
        lam_pref: float,
        normalize_advantage: bool = True,
    ) -> None:
        """Two independent GAE passes with their own discounts, normalised separately.

        ``gamma_pref << gamma`` is expected to help: manner is mostly local while success is
        long-horizon. It is a configurable ablation, not a hidden default.
        """
        n = self.num_transitions_per_env

        self.returns, self.advantages = self._gae(
            self.rewards, self.values, self.dones, last_values, gamma, lam, n
        )
        self.returns_pref, self.advantages_pref = self._gae(
            self.rewards_pref, self.values_pref, self.dones, last_values_pref, gamma_pref, lam_pref, n
        )

        if normalize_advantage:
            self.advantages = self._normalize(self.advantages)
            # Guarded: with a zero preference reward (sanity run A) the advantages are identically
            # zero, and dividing by their std would turn 0/1e-8 into noise. Leave them at zero.
            self.advantages_pref = self._normalize(self.advantages_pref)

    @staticmethod
    def _normalize(adv: torch.Tensor) -> torch.Tensor:
        std = adv.std()
        if not torch.isfinite(std) or std < 1e-8:
            return torch.zeros_like(adv)
        return (adv - adv.mean()) / (std + 1e-8)

    # -- minibatches -------------------------------------------------------------------------

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        """Same as upstream, plus the preference target values / advantages / returns.

        Reimplemented rather than wrapped because the upstream generator draws its own random
        permutation; both streams must be indexed by the *same* permutation.
        """
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        values_pref = self.values_pref.flatten(0, 1)
        returns_pref = self.returns_pref.flatten(0, 1)
        advantages_pref = self.advantages_pref.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                batch_idx = indices[i * mini_batch_size : (i + 1) * mini_batch_size]
                yield (
                    observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                    # preference stream
                    values_pref[batch_idx],
                    advantages_pref[batch_idx],
                    returns_pref[batch_idx],
                )

    def recurrent_mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8) -> Generator:
        raise NotImplementedError(
            "DualRolloutStorage does not support recurrent policies. The OmniReset agent config uses "
            "a feedforward ActorCritic (gSDE noise), so this path is unused; implement it if that changes."
        )
