# Copyright (c) 2026, Null-space preference critic project.
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-critic with two value heads: one for task return, one for preference return.

Two architectures, both worth measuring (see design notes §3.2):

``shared``   one critic trunk, final layer widened 1 -> 2 outputs. Cheapest, and matches the
             "shared trunk, second output head" of the design. Caveat: the preference value loss
             backpropagates through the shared trunk, so the two heads are coupled *even at β=0*.
             That coupling is exactly what sanity run A measures.
``separate`` a second independent critic MLP. Strictly more parameters, but zero coupling at
             β=0, so the β=0 overlay is guaranteed by construction rather than by experiment.

The base class' ``evaluate`` keeps returning the *task* value with shape (N, 1) so that any
untouched upstream code path behaves exactly as before.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import Any

from rsl_rl.modules import ActorCritic
from rsl_rl.networks import MLP

CRITIC_ARCHES = ("shared", "separate")


class DualCriticActorCritic(ActorCritic):
    """Asymmetric actor-critic (critic sees privileged obs) with task and preference value heads."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        critic_arch: str = "shared",
        **kwargs: dict[str, Any],
    ) -> None:
        if critic_arch not in CRITIC_ARCHES:
            raise ValueError(f"critic_arch must be one of {CRITIC_ARCHES}, got {critic_arch!r}")

        super().__init__(
            obs,
            obs_groups,
            num_actions,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            **kwargs,
        )

        self.critic_arch = critic_arch

        num_critic_obs = sum(obs[g].shape[-1] for g in obs_groups["critic"])

        if critic_arch == "shared":
            # Replace the 1-output critic built by the base class with a 2-output one.
            self.critic = MLP(num_critic_obs, 2, critic_hidden_dims, activation)
            print(f"Critic MLP (shared trunk, 2 heads): {self.critic}")
        else:
            self.critic_pref = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
            print(f"Critic MLP (separate preference critic): {self.critic_pref}")

    # -- parameter partition -------------------------------------------------------------------
    # The projection is applied to the *actor* gradient only; the value losses train the critics
    # normally. EmpiricalNormalization holds buffers, not parameters, so a name-prefix split is
    # exact here.

    #: Parameter names that scale exploration noise rather than the deployed mean action.
    NOISE_PARAM_NAMES = ("std", "log_std")

    def actor_parameters(self) -> list[torch.nn.Parameter]:
        """Actor MLP plus the exploration-noise parameters (``std`` / ``log_std`` for gSDE)."""
        return [p for n, p in self.named_parameters() if not n.startswith("critic")]

    def critic_parameters(self) -> list[torch.nn.Parameter]:
        return [p for n, p in self.named_parameters() if n.startswith("critic")]

    def is_noise_param_name(self, name: str) -> bool:
        return name.split(".")[0] in self.NOISE_PARAM_NAMES

    def actor_param_noise_mask(self) -> list[bool]:
        """Per-entry flags over ``actor_parameters()``: True where the parameter is noise scale.

        The preference objective is a statement about *deployed behaviour*, and the policy is
        deployed on the mean action -- gSDE noise does not exist at deployment time. So the
        preference gradient has no business touching these.
        """
        return [self.is_noise_param_name(n) for n, _ in self.named_parameters() if not n.startswith("critic")]

    # -- preference-scoped log probability ------------------------------------------------------

    def log_prob_mean_path(self, obs: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """Log-prob whose gradient reaches the network **only through the mean action**.

        Numerically identical to the usual log-prob -- ``detach`` changes no values -- so the PPO
        importance ratio built from it is the same number, and "one ratio, one clip" still holds.
        Only the backward graph differs.

        Why this is needed (Layer 2): gSDE's action variance is
        ``mm(features**2, exp(log_std)**2)`` where ``features = actor[:-1](obs)``
        (rsl_rl/modules/actor_critic.py:72, :283). Excluding ``log_std`` from the preference
        gradient (Layer 1) therefore does *not* close the path -- the preference can still shrink
        exploration by reshaping the trunk features that feed the noise head. Detaching both
        inputs to the variance closes it.
        """
        obs = self.get_actor_obs(obs)
        obs = self.actor_obs_normalizer(obs)
        mean = self.actor(obs)

        if self.noise_std_type == "gsde":
            features = self.actor[:-1](obs).detach()
            std = torch.sqrt(
                torch.mm(features**2, torch.exp(self.log_std.detach()) ** 2) + self.distribution.epsilon
            )
        elif self.noise_std_type == "scalar":
            std = self.std.detach().expand_as(mean)
        else:  # "log"
            std = torch.exp(self.log_std.detach()).expand_as(mean)

        return torch.distributions.Normal(mean, std).log_prob(actions).sum(dim=-1)

    # -- diagnostics ---------------------------------------------------------------------------

    def noise_magnitude(self) -> torch.Tensor:
        """Mean realised action std of the current distribution (the exploration-leak canary)."""
        return self.action_std.mean().detach()

    @torch.no_grad()
    def noise_decomposition(self, obs: TensorDict) -> tuple[float, float]:
        """Split realised noise into its two multiplicative factors.

        gSDE variance is ``mm(f(obs)**2, exp(log_std)**2)`` with ``f = actor[:-1]``, so

            log(realised std)  ≈  log‖f(obs)‖  +  log σ
                                   trunk path      direct path
                                   (Layer 2)       (Layer 1)

        Layer 1 masks **only the second factor**, so an aggregate fall in realised noise cannot
        distinguish "the trunk reshaped its features" from "sigma shrank". Reporting them
        separately makes the Layer 2 decision mechanical:

          sigma flat, ‖f‖ flat        -> no leak
          sigma flat, ‖f‖ collapsing  -> trunk path, Layer 2 indicated
          sigma moving at all         -> Layer 1 implementation bug (it is excluded from g_pref
                                         by construction, so nothing else can move it)

        Returns ``(mean ‖f(obs)‖, mean sigma)``.
        """
        if self.noise_std_type != "gsde":
            std = self.std if self.noise_std_type == "scalar" else torch.exp(self.log_std)
            return 1.0, float(std.mean())
        x = self.actor_obs_normalizer(self.get_actor_obs(obs))
        feat = self.actor[:-1](x)
        return float(feat.norm(dim=-1).mean()), float(torch.exp(self.log_std).mean())

    # -- value heads ---------------------------------------------------------------------------

    def _critic_features(self, obs: TensorDict) -> torch.Tensor:
        obs = self.get_critic_obs(obs)
        return self.critic_obs_normalizer(obs)

    def evaluate_dual(self, obs: TensorDict, **kwargs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(value_task, value_pref)``, each shaped (N, 1)."""
        x = self._critic_features(obs)
        if self.critic_arch == "shared":
            out = self.critic(x)
            return out[..., 0:1], out[..., 1:2]
        return self.critic(x), self.critic_pref(x)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        """Task value only, shape (N, 1) -- preserves the upstream contract."""
        x = self._critic_features(obs)
        if self.critic_arch == "shared":
            return self.critic(x)[..., 0:1]
        return self.critic(x)
