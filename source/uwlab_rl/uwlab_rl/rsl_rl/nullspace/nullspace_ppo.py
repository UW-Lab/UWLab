# Copyright (c) 2026, Null-space preference critic project.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO with a dual critic and a null-space-projected preference gradient.

Structure of one update, and why each piece is where it is:

* **One importance ratio, one clip.** Upstream already computes ``ratio`` once; both the task and
  preference surrogates are built from that same clipped ratio. This is not two PPO updates.
* **The projection is applied to gradients, not advantages.** The first-order guarantee in the
  design is a statement about ``g_task`` and ``g_pref``; combining advantages before the surrogate
  is a per-sample reweighting that does *not* satisfy it. Phase 0 measured the PPO update at
  ~1% of iteration wall-clock (collection dominates ~99%), so the extra backward costs ~1%.
  ``projection_mode="advantage"`` implements the cheap approximation as an ablation.
* **Separately normalised advantages** (in :class:`DualRolloutStorage`) are what make β
  dimensionless and independent of the preference reward's scale.
"""

from __future__ import annotations

import importlib.metadata as metadata
import inspect
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms import PPO

from .dual_storage import DualRolloutStorage
from .projection import flatten_grads, project_nullspace, unflatten_to

PROJECTION_MODES = ("gradient", "advantage", "sum")


class NullspacePPO(PPO):
    """PPO whose preference objective is projected into the null space of the task gradient."""

    def __init__(
        self,
        policy,  # noqa: ANN001 -- DualCriticActorCritic
        beta: float = 0.0,
        gamma_pref: float | None = None,
        lam_pref: float | None = None,
        projection_mode: str = "gradient",
        pref_value_loss_coef: float | None = None,
        **kwargs,
    ) -> None:
        # IsaacLab's RslRlPpoAlgorithmCfg carries fields that only exist in rsl-rl >= 4.x
        # (e.g. `share_cnn_encoders`, `optimizer`). Upstream's `sanitize_rsl_rl_cfg` strips those
        # by resolving the class out of `rsl_rl.algorithms` -- which cannot find NullspacePPO,
        # so they arrive here intact and PPO 3.1.2 rejects them. Drop them explicitly, loudly.
        accepted = set(inspect.signature(PPO.__init__).parameters) - {"self", "policy"}
        dropped = sorted(set(kwargs) - accepted)
        if dropped:
            print(
                f"[NullspacePPO] Dropping algorithm config keys unsupported by the installed "
                f"rsl-rl ({metadata.version('rsl-rl-lib')}): {dropped}"
            )
        super().__init__(policy, **{k: v for k, v in kwargs.items() if k in accepted})

        if projection_mode not in PROJECTION_MODES:
            raise ValueError(f"projection_mode must be one of {PROJECTION_MODES}, got {projection_mode!r}")
        if self.symmetry is not None:
            raise NotImplementedError("NullspacePPO does not support symmetry augmentation.")
        if self.rnd is not None:
            raise NotImplementedError(
                "NullspacePPO does not support RND. RND folds a second reward stream into a single "
                "critic by weighted sum -- that is the baseline this method is compared against."
            )
        if not hasattr(policy, "evaluate_dual"):
            raise TypeError("NullspacePPO requires a DualCriticActorCritic (missing evaluate_dual).")

        self.beta = beta
        # Manner is mostly local, success is long-horizon; default to matching the task discount so
        # that any difference is an explicit experimental choice rather than a hidden default.
        self.gamma_pref = self.gamma if gamma_pref is None else gamma_pref
        self.lam_pref = self.lam if lam_pref is None else lam_pref
        self.projection_mode = projection_mode
        self.pref_value_loss_coef = (
            self.value_loss_coef if pref_value_loss_coef is None else pref_value_loss_coef
        )

        self.transition = DualRolloutStorage.Transition()
        self._diag: dict[str, float] = {}

    # -- storage ---------------------------------------------------------------------------

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape) -> None:  # noqa: ANN001
        self.storage = DualRolloutStorage(
            training_type, num_envs, num_transitions_per_env, obs, actions_shape, self.device
        )

    # -- rollout ---------------------------------------------------------------------------

    def act(self, obs: TensorDict) -> torch.Tensor:
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()
        self.transition.actions = self.policy.act(obs).detach()
        # One critic forward for both heads.
        v_task, v_pref = self.policy.evaluate_dual(obs)
        self.transition.values = v_task.detach()
        self.transition.values_pref = v_pref.detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict
    ) -> None:
        """Record a transition.

        Reimplemented rather than delegated because the time-out bootstrap must be applied **per
        stream**, each with its own discount and value head. Episodes here are 16 s / 160 control
        steps and ``time_out`` is the dominant termination, so this path fires constantly -- using
        the task discount for the preference return would be wrong at every episode boundary.
        """
        self.policy.update_normalization(obs)

        rewards_pref = extras.get("reward_pref")
        if rewards_pref is None:
            raise KeyError(
                "extras['reward_pref'] missing. Wrap the env in DualRewardVecEnvWrapper so the "
                "preference stream is emitted alongside the task reward."
            )

        self.transition.rewards = rewards.clone()
        self.transition.rewards_pref = rewards_pref.clone().to(self.device)
        self.transition.dones = dones

        if "time_outs" in extras:
            time_outs = extras["time_outs"].unsqueeze(1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * time_outs, 1)
            self.transition.rewards_pref += self.gamma_pref * torch.squeeze(
                self.transition.values_pref * time_outs, 1
            )

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        with torch.no_grad():
            last_v_task, last_v_pref = self.policy.evaluate_dual(obs)
        self.storage.compute_returns_dual(
            last_v_task.detach(),
            last_v_pref.detach(),
            gamma=self.gamma,
            lam=self.lam,
            gamma_pref=self.gamma_pref,
            lam_pref=self.lam_pref,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    # -- update ----------------------------------------------------------------------------

    def _surrogate(self, advantages: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
        """PPO clipped surrogate loss for one advantage stream, sharing the given ratio."""
        adv = torch.squeeze(advantages)
        surrogate = -adv * ratio
        surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        return torch.max(surrogate, surrogate_clipped).mean()

    def _value_loss(self, value: torch.Tensor, target: torch.Tensor, returns: torch.Tensor) -> torch.Tensor:
        if not self.use_clipped_value_loss:
            return (returns - value).pow(2).mean()
        clipped = target + (value - target).clamp(-self.clip_param, self.clip_param)
        return torch.max((value - returns).pow(2), (clipped - returns).pow(2)).mean()

    def update(self) -> dict[str, float]:
        stats = {
            "value_function": 0.0,
            "value_function_pref": 0.0,
            "surrogate": 0.0,
            "surrogate_pref": 0.0,
            "entropy": 0.0,
        }
        diag_sums: dict[str, float] = {}

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        actor_params = self.policy.actor_parameters()

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hidden_states_batch,
            masks_batch,
            target_values_pref_batch,
            advantages_pref_batch,
            returns_pref_batch,
        ) in generator:

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = self._norm(advantages_batch)
                    advantages_pref_batch = self._norm(advantages_pref_batch)

            self.policy.act(obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch, value_pref_batch = self.policy.evaluate_dual(
                obs_batch, masks=masks_batch, hidden_state=hidden_states_batch[1]
            )
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            self._adapt_learning_rate(mu_batch, sigma_batch, old_mu_batch, old_sigma_batch)

            # One ratio, one clip, shared by both objectives.
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate_task = self._surrogate(advantages_batch, ratio)
            surrogate_pref = self._surrogate(advantages_pref_batch, ratio)

            value_loss = self._value_loss(value_batch, target_values_batch, returns_batch)
            value_loss_pref = self._value_loss(value_pref_batch, target_values_pref_batch, returns_pref_batch)

            # Critic losses + entropy bonus. The surrogates are handled separately below because
            # their gradients must be projected before they reach the actor.
            loss_rest = (
                self.value_loss_coef * value_loss
                + self.pref_value_loss_coef * value_loss_pref
                - self.entropy_coef * entropy_batch.mean()
            )

            self.optimizer.zero_grad()

            if self.projection_mode == "gradient":
                actor_grad, diag = self._projected_actor_grad(surrogate_task, surrogate_pref, actor_params)
                loss_rest.backward()
                for p, g in zip(actor_params, unflatten_to(actor_grad, actor_params)):
                    p.grad = g.clone() if p.grad is None else p.grad + g
            else:
                # Ablations: combine the scalar objectives instead of their gradients.
                # "sum" is the weighted-sum baseline (arm B); "advantage" is the cheap
                # approximation of the projection that drops the first-order guarantee.
                combined = surrogate_task + self.beta * surrogate_pref
                (combined + loss_rest).backward()
                diag = {}

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            stats["value_function"] += value_loss.item()
            stats["value_function_pref"] += value_loss_pref.item()
            stats["surrogate"] += surrogate_task.item()
            stats["surrogate_pref"] += surrogate_pref.item()
            stats["entropy"] += entropy_batch.mean().item()
            for k, v in diag.items():
                diag_sums[k] = diag_sums.get(k, 0.0) + v

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in stats:
            stats[k] /= num_updates
        for k, v in diag_sums.items():
            stats[f"proj/{k}"] = v / num_updates
        stats["beta"] = self.beta

        # Per-stream reward means. The runner's reward bookkeeping only tracks the scalar it gets
        # from env.step (the task stream), so log the preference stream here -- attribution
        # between the two curves is the point of keeping them separate (design notes §3.4).
        stats["reward_task_mean"] = self.storage.rewards.mean().item()
        stats["reward_pref_mean"] = self.storage.rewards_pref.mean().item()

        self.storage.clear()
        self._diag = stats
        return stats

    # -- helpers ---------------------------------------------------------------------------

    def _projected_actor_grad(
        self, surrogate_task: torch.Tensor, surrogate_pref: torch.Tensor, actor_params: list
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Flat actor gradient with the preference component projected orthogonally to the task."""
        g_task = flatten_grads(
            torch.autograd.grad(surrogate_task, actor_params, retain_graph=True, allow_unused=True),
            actor_params,
        )
        if self.beta == 0.0:
            # Skip the second backward entirely: at β=0 the projection is the identity, so this
            # path is exactly baseline PPO plus an untouched second critic head.
            return g_task, {"g_task_norm": torch.linalg.vector_norm(g_task).item()}

        g_pref = flatten_grads(
            torch.autograd.grad(surrogate_pref, actor_params, retain_graph=True, allow_unused=True),
            actor_params,
        )
        return project_nullspace(g_task, g_pref, self.beta)

    def _adapt_learning_rate(self, mu, sigma, old_mu, old_sigma) -> None:  # noqa: ANN001
        """Unchanged from upstream: KL-adaptive LR on the *policy* distribution only."""
        if self.desired_kl is None or self.schedule != "adaptive":
            return
        with torch.inference_mode():
            kl = torch.sum(
                torch.log(sigma / old_sigma + 1.0e-5)
                + (torch.square(old_sigma) + torch.square(old_mu - mu)) / (2.0 * torch.square(sigma))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)
            if self.is_multi_gpu:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= self.gpu_world_size
            if self.gpu_global_rank == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif self.desired_kl / 2.0 > kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            if self.is_multi_gpu:
                lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                torch.distributed.broadcast(lr_tensor, src=0)
                self.learning_rate = lr_tensor.item()
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

    @staticmethod
    def _norm(adv: torch.Tensor) -> torch.Tensor:
        std = adv.std()
        if not torch.isfinite(std) or std < 1e-8:
            return torch.zeros_like(adv)
        return (adv - adv.mean()) / (std + 1e-8)
