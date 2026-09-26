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
from rsl_rl.networks import EmpiricalDiscountedVariationNormalization

from .dual_storage import DualRolloutStorage
from .projection import (
    clip_grad_norm_by_group,
    flat_mask,
    flatten_grads,
    partition_policy_params,
    project_nullspace,
    project_nullspace_masked,
    unflatten_to,
)

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
        pref_mask_noise: bool = True,
        pref_detach_noise_features: bool = False,
        log_pref_alignment: bool = True,
        grad_clip_mode: str = "per_group",
        normalize_pref_reward: bool = True,
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

        # --- exploration-leak scoping -----------------------------------------------------
        # Every scripted manner preference (mechanical power, EE speed, action smoothness) is
        # monotonically improved by shrinking action noise, while perturbing noise around a local
        # optimum of the mean policy is ~second order in task return. Large first-order preference
        # gradient against ~zero first-order task gradient means the projector does not merely
        # permit that direction -- it *selects* it. That is entropy collapse arriving through the
        # exact channel built to find task-neutral directions.
        #
        # gSDE noise is an optimisation parameter, not a deployed behavioural property (the policy
        # is deployed on the mean action), so no legitimate preference is given up by masking it.
        self.pref_mask_noise = pref_mask_noise  # Layer 1: exclude noise params from g_pref
        self.pref_detach_noise_features = pref_detach_noise_features  # Layer 2: cut the trunk path
        self._pref_allowed: torch.Tensor | None = None  # lazily built flat subspace mask
        # pref_removed_frac is the screening instrument for preference/task conflict, so it is
        # logged in EVERY arm as a time series -- including beta=0, where nothing is applied.
        self.log_pref_alignment = log_pref_alignment

        # --- gradient-norm clipping scope (NOTES 29) ----------------------------------------
        # Upstream clips the whole policy as one vector. For a dual critic that couples the
        # preference critic to the actor: its value loss enters the same norm, and a large one
        # scales the actor's update down -- at beta=0 too. Clip each group on its own budget.
        if grad_clip_mode not in ("per_group", "global"):
            raise ValueError(f"grad_clip_mode must be 'per_group' or 'global', got {grad_clip_mode!r}")
        self.grad_clip_mode = grad_clip_mode

        # --- preference reward scale (NOTES 29) ---------------------------------------------
        # The preference critic regresses onto returns of an arbitrary-scale reward; with
        # action_rate its loss reached 1e5-1e6. A positive running scale leaves the normalised
        # preference advantages unchanged, so this fixes the critic without changing what the
        # actor sees. Registered on the policy so its running stats are checkpointed with it.
        # The task stream is deliberately NOT normalised.
        self.normalize_pref_reward = normalize_pref_reward
        if normalize_pref_reward:
            self.policy.pref_reward_normalizer = EmpiricalDiscountedVariationNormalization(
                shape=1, gamma=self.gamma_pref
            ).to(self.device)

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
        if self.normalize_pref_reward:
            # Before the time-out bootstrap below: that adds gamma_pref * values_pref, and the
            # preference critic's values live in this normalised scale once it trains on it.
            self.transition.rewards_pref = self.policy.pref_reward_normalizer(
                self.transition.rewards_pref.view(-1, 1)
            ).view(-1)
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
            # Guard metrics -- logged in EVERY arm including beta=0, because the exploration leak
            # is only visible as drift *relative to the beta=0 reference*.
            "guard_entropy": 0.0,
            "guard_noise_std": 0.0,
            # Decomposition of realised noise into its two multiplicative factors. Layer 1 masks
            # only `sigma`; a fall in the aggregate cannot say which path moved.
            "guard_feat_norm": 0.0,
            "guard_sigma": 0.0,
        }
        diag_sums: dict[str, float] = {}

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        actor_params = self.policy.actor_parameters()
        param_groups = partition_policy_params(self.policy.named_parameters())
        grad_norm_sums: dict[str, float] = {}

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

            if self.pref_detach_noise_features:
                # Layer 2: same ratio *value*, different backward graph -- the preference
                # objective reaches the network only through the mean action, so it cannot be
                # satisfied by reshaping the trunk features that set the gSDE noise scale.
                lp_pref = self.policy.log_prob_mean_path(obs_batch, actions_batch)
                ratio_pref = torch.exp(lp_pref - torch.squeeze(old_actions_log_prob_batch))
                self._assert_ratio_equivalence(ratio, ratio_pref)
            else:
                ratio_pref = ratio
            surrogate_pref = self._surrogate(advantages_pref_batch, ratio_pref)

            value_loss = self._value_loss(value_batch, target_values_batch, returns_batch)
            value_loss_pref = self._value_loss(value_pref_batch, target_values_pref_batch, returns_pref_batch)

            # Critic losses + entropy bonus. The surrogates are handled separately below because
            # their gradients must be projected before they reach the actor.
            #
            # NOTE: the entropy bonus deliberately sits HERE and not inside either surrogate. It
            # is a regulariser on the optimisation, not a preference about behaviour, so it stays
            # on the task side of the split -- it must keep its unrestricted gradient path to the
            # noise parameters even when the preference term is masked off them.
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
            elif self.projection_mode == "sum":
                # ---- Arm B′: dual critic, separately normalised advantages, NO projection ----
                # This isolates the two contributions the design claims separately: scale
                # invariance (from per-stream advantage normalisation) and the projection itself.
                # Without B′ the Pareto plot cannot tell them apart -- if C ties B′, the
                # contribution is scale-free preference control and the projection is a safety
                # belt, which is a different claim from the one currently drafted.
                # Diagnostics are still computed so pref_removed_frac remains comparable.
                diag = self._alignment_diagnostics(surrogate_task, surrogate_pref, actor_params)
                (surrogate_task + self.beta * surrogate_pref + loss_rest).backward()
            else:  # "advantage"
                # Cheap approximation: combine the per-sample advantages *before* the surrogate.
                # This is a per-sample reweighting and does NOT satisfy the first-order guarantee.
                adv = torch.squeeze(advantages_batch) + self.beta * torch.squeeze(advantages_pref_batch)
                surrogate_combined = self._surrogate(adv.unsqueeze(-1), ratio)
                diag = self._alignment_diagnostics(surrogate_task, surrogate_pref, actor_params)
                (surrogate_combined + loss_rest).backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            if self.grad_clip_mode == "per_group":
                gnorms = clip_grad_norm_by_group(param_groups, self.max_grad_norm)
            else:  # "global": upstream behaviour, kept only to reproduce pre-fix runs (NOTES 29)
                total = nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                gnorms = {"global": float(total)}
            self.optimizer.step()
            for k, v in gnorms.items():
                grad_norm_sums[k] = grad_norm_sums.get(k, 0.0) + v

            stats["value_function"] += value_loss.item()
            stats["value_function_pref"] += value_loss_pref.item()
            stats["surrogate"] += surrogate_task.item()
            stats["surrogate_pref"] += surrogate_pref.item()
            stats["entropy"] += entropy_batch.mean().item()
            stats["guard_entropy"] += entropy_batch.mean().item()
            stats["guard_noise_std"] += self.policy.noise_magnitude().item()
            fnorm, sigma = self.policy.noise_decomposition(obs_batch)
            stats["guard_feat_norm"] += fnorm
            stats["guard_sigma"] += sigma
            for k, v in diag.items():
                diag_sums[k] = diag_sums.get(k, 0.0) + v

        num_updates = self.num_learning_epochs * self.num_mini_batches
        for k in stats:
            stats[k] /= num_updates
        for k, v in diag_sums.items():
            stats[f"proj/{k}"] = v / num_updates
        # Pre-clip gradient norm per group: makes a cross-group throttle directly visible
        # instead of inferring it from the adaptive learning rate (NOTES 29).
        for k, v in grad_norm_sums.items():
            stats[f"grad_norm/{k}"] = v / num_updates
        stats["beta"] = self.beta
        stats["pref_mask_noise"] = float(self.pref_mask_noise)
        stats["pref_detach_noise_features"] = float(self.pref_detach_noise_features)
        stats["grad_clip_per_group"] = float(self.grad_clip_mode == "per_group")
        stats["normalize_pref_reward"] = float(self.normalize_pref_reward)
        if self.normalize_pref_reward:
            stats["pref_reward_scale"] = float(self.policy.pref_reward_normalizer.emp_norm._std)

        # Per-stream reward means. The runner's reward bookkeeping only tracks the scalar it gets
        # from env.step (the task stream), so log the preference stream here -- attribution
        # between the two curves is the point of keeping them separate (design notes §3.4).
        stats["reward_task_mean"] = self.storage.rewards.mean().item()
        stats["reward_pref_mean"] = self.storage.rewards_pref.mean().item()

        self.storage.clear()
        self._diag = stats
        return stats

    # -- helpers ---------------------------------------------------------------------------

    def _preference_subspace(self, actor_params: list) -> torch.Tensor | None:
        """Flat mask of the coordinates the preference term may move (None = all of them)."""
        if not self.pref_mask_noise:
            return None
        if self._pref_allowed is None:
            noise_flags = self.policy.actor_param_noise_mask()
            self._pref_allowed = ~flat_mask(actor_params, noise_flags)
        return self._pref_allowed

    def _alignment_diagnostics(
        self, surrogate_task: torch.Tensor, surrogate_pref: torch.Tensor, actor_params: list
    ) -> dict[str, float]:
        """Measure task/preference gradient alignment **without applying** the preference.

        Logged in every arm -- including β=0 and the unprojected ablations -- because
        ``pref_removed_frac`` is the screening instrument for whether a preference genuinely
        conflicts with the task, and it must be a *time series*: alignment moves over training,
        and the late-training regime (g_task → 0, projector → identity) is where the
        self-scheduling argument lives. A single converged number is not evidence for that.

        Costs one extra backward (~1% of iteration wall-clock, since collection dominates ~99%).
        The gradient actually applied is unaffected.
        """
        if not self.log_pref_alignment:
            return {}
        g_task = flatten_grads(
            torch.autograd.grad(surrogate_task, actor_params, retain_graph=True, allow_unused=True),
            actor_params,
        )
        g_pref = flatten_grads(
            torch.autograd.grad(surrogate_pref, actor_params, retain_graph=True, allow_unused=True),
            actor_params,
        )
        allowed = self._preference_subspace(actor_params)
        if allowed is not None:
            g_task, g_pref = g_task[allowed], g_pref[allowed]
        # beta=1 purely to read out the geometry; nothing is applied.
        _, diag = project_nullspace(g_task, g_pref, beta=1.0)
        return diag

    def _projected_actor_grad(
        self, surrogate_task: torch.Tensor, surrogate_pref: torch.Tensor, actor_params: list
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Flat actor gradient with the preference component projected orthogonally to the task."""
        g_task = flatten_grads(
            torch.autograd.grad(surrogate_task, actor_params, retain_graph=True, allow_unused=True),
            actor_params,
        )
        if self.beta == 0.0:
            # At β=0 the projection is the identity, so the APPLIED gradient is exactly baseline
            # PPO plus an untouched second critic head. The alignment read-out is still taken
            # (extra backward, nothing applied) so the β=0 reference has the same time series as
            # every other arm -- otherwise there is nothing to compare the others against.
            diag = self._alignment_diagnostics(surrogate_task, surrogate_pref, actor_params)
            diag.setdefault("g_task_norm", torch.linalg.vector_norm(g_task).item())
            diag["pref_contrib_ratio"] = 0.0
            return g_task, diag

        g_pref = flatten_grads(
            torch.autograd.grad(surrogate_pref, actor_params, retain_graph=True, allow_unused=True),
            actor_params,
        )

        allowed = self._preference_subspace(actor_params)
        if allowed is None:
            # Unmasked: the known-degenerate configuration. Kept runnable on purpose so the
            # exploration collapse can be demonstrated once rather than argued about.
            return project_nullspace(g_task, g_pref, self.beta)

        combined, diag = project_nullspace_masked(g_task, g_pref, self.beta, allowed)
        self._assert_noise_params_untouched(combined, g_task, allowed)
        return combined, diag

    def _assert_noise_params_untouched(
        self, combined: torch.Tensor, g_task: torch.Tensor, allowed: torch.Tensor
    ) -> None:
        """Layer 1's claim, checked directly: preference contributes EXACTLY zero to noise params.

        Under Layer 1 masking, ``sigma`` is excluded from ``g_pref`` by construction, so nothing
        in the preference path can move it. If it moves, that is an implementation bug -- not a
        leak -- and the two must never be confused when reading the guard metrics. Asserting it
        here is stronger than inferring flatness from a cross-run plot.
        """
        if getattr(self, "_mask_checked", False):
            return
        self._mask_checked = True
        excluded = ~allowed
        if excluded.any() and not torch.equal(combined[excluded], g_task[excluded]):
            d = (combined[excluded] - g_task[excluded]).abs().max().item()
            raise RuntimeError(
                f"Layer 1 masking violated: preference moved the noise parameters "
                f"(max |diff| {d:.3e}). The projection must be *restricted* to the mean-action "
                "subspace, not applied to a zero-padded g_pref."
            )
        n_excluded = int(excluded.sum())
        print(
            f"[NullspacePPO] Layer 1 active: {n_excluded} noise params "
            f"({100 * n_excluded / excluded.numel():.2f}% of actor) excluded from g_pref; verified untouched."
        )

    def _assert_ratio_equivalence(self, ratio: torch.Tensor, ratio_pref: torch.Tensor) -> None:
        """Check once that Layer 2 changed only the gradient path, not the objective's value.

        If these ever differ numerically we are no longer running "one ratio, one clip" -- we are
        running two different PPO objectives, and the clipping would disagree between streams.
        """
        if getattr(self, "_ratio_checked", False):
            return
        self._ratio_checked = True
        if not torch.allclose(ratio.detach(), ratio_pref.detach(), rtol=1e-4, atol=1e-6):
            d = (ratio - ratio_pref).abs().max().item()
            raise RuntimeError(
                f"Layer-2 preference ratio diverged from the task ratio (max |diff| {d:.3e}). "
                "log_prob_mean_path must reproduce the policy's log-prob exactly and differ only "
                "in its backward graph."
            )
        print("[NullspacePPO] Layer 2 active: preference ratio value-identical, mean-path gradient only.")

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
