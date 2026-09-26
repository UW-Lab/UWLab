# Copyright (c) 2026, Null-space preference critic project.
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the dual-critic / null-space-projection PPO agent."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from .rl_cfg import RslRlFancyActorCriticCfg


@configclass
class RslRlNullspaceActorCriticCfg(RslRlFancyActorCriticCfg):
    """Actor-critic with a second value head for the preference return."""

    class_name: str = "DualCriticActorCritic"

    critic_arch: str = "shared"
    """``shared``: one critic trunk widened to 2 outputs (cheap; the preference value loss
    backpropagates through the shared trunk, so the heads are coupled even at beta=0).
    ``separate``: an independent preference critic MLP -- no coupling through shared weights.
    That alone does NOT guarantee the beta=0 overlay: until ``grad_clip_mode='per_group'`` the
    actor and both critics shared one gradient-norm clip, which coupled them regardless of
    architecture (NOTES 29). Run sanity check A against both."""


@configclass
class RslRlNullspacePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO whose preference gradient is projected into the null space of the task gradient."""

    class_name: str = "NullspacePPO"

    beta: float = 0.0
    """Preference step budget. 0.0 applies no preference gradient to the actor.

    It reproduced baseline PPO only with ``pref_source='zero'``. Under a non-zero preference the
    preference critic's gradient reached the actor through the shared gradient-norm clip, so
    beta=0 was not a no-op (NOTES 29). With ``grad_clip_mode='per_group'`` it is, for any
    ``pref_source``."""

    gamma_pref: float | None = None
    """Discount for the preference return. None mirrors ``gamma``. Manner is mostly local while
    success is long-horizon, so gamma_pref << gamma is expected to help -- kept as an explicit
    ablation rather than a hidden default."""

    lam_pref: float | None = None
    """GAE lambda for the preference stream. None mirrors ``lam``."""

    projection_mode: str = "gradient"
    """``gradient``: project g_pref orthogonally to g_task (faithful; costs one extra backward,
    measured at ~1% of iteration wall-clock since collection dominates).
    ``advantage``/``sum``: combine the scalar objectives instead -- the weighted-sum baseline and
    the cheap approximation that drops the first-order guarantee. Ablations only."""

    pref_value_loss_coef: float | None = None
    """Value-loss coefficient for the preference critic. None mirrors ``value_loss_coef``."""

    pref_mask_noise: bool = True
    """Layer 1. Exclude the exploration-noise parameters (gSDE ``log_std``) from the preference
    gradient, and run the projection inside the remaining (mean-action) subspace.

    Default ON. Every scripted manner preference is monotonically improved by shrinking action
    noise, while noise perturbations are ~second order in task return near a local optimum -- so
    an unmasked projector actively *selects* exploration collapse as a "task-neutral" direction.
    gSDE noise does not exist at deployment (the policy is deployed on the mean action), so
    masking gives up no legitimate preference. Set False only for the deliberate demonstration
    run of the degenerate mode."""

    pref_detach_noise_features: bool = False
    """Layer 2 escalation. Also detach the trunk features that feed the gSDE noise head inside
    the preference surrogate.

    Layer 1 alone does NOT close the leak: the noise scale is
    ``mm(actor[:-1](obs)**2, exp(log_std)**2)``, so the preference can still shrink exploration by
    reshaping the trunk. Enable if guard entropy / noise magnitude drift against the beta=0
    reference after Layer 1."""

    grad_clip_mode: str = "per_group"
    """How ``max_grad_norm`` is applied. ``per_group`` clips the actor, the task critic and the
    preference critic each on its own budget; ``global`` clips them as one vector (upstream).

    ``global`` is a bug for a dual-critic agent and is kept only to reproduce pre-fix runs
    (NOTES 29). The preference critic's gradient enters the same norm as the actor's, so a large
    preference value loss scales the actor's update down -- at beta=0 as well. With
    ``pref_source='action_rate'`` that loss reached 1e5-1e6 against ``max_grad_norm=1.0``: the
    actor was throttled, the adaptive schedule pinned the learning rate at its 1e-2 cap, and
    because the preference scales with action noise the throttle tracked exploration."""

    normalize_pref_reward: bool = True
    """Scale the preference reward by a running std of its discounted sum (Pathak et al.; rsl_rl's
    ``EmpiricalDiscountedVariationNormalization``) before it reaches the preference critic.

    Per-stream advantage normalisation already makes the *actor's* preference signal scale-free,
    but the preference critic regressed onto raw returns, so its loss carried the reward's
    arbitrary scale. A positive running factor leaves the normalised preference advantages
    unchanged, so this fixes the critic without altering what the actor sees. The task stream is
    never normalised: it must stay identical to upstream for the beta=0 null to mean anything."""


@configclass
class RslRlNullspaceRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner that can resolve the null-space classes by name."""

    class_name: str = "DualCriticOnPolicyRunner"

    pref_source: str = "zero"
    """Preference reward stream:
    ``zero``            sanity run A;
    ``noise``           sanity run B -- an *uninformative* preference the projection must not
                        amplify into variance;
    ``action_rate``     the noise-bait probe -- a preference that is maximally satisfiable by
                        shrinking exploration and barely satisfiable any other way. Correctly
                        scoped it should yield almost NO compliance gain; large apparent gain
                        means the exploration leak is still open;
    ``terms``           Phase 2 scripted manner predicates read out of the RewardManager."""

    pref_noise_std: float = 1.0
    """Std of the Gaussian preference reward when ``pref_source='noise'``."""

    pref_term_names: tuple[str, ...] = ()
    """RewardManager term names forming the preference stream when ``pref_source='terms'``.
    These stay in the manager (evaluation order untouched) but are subtracted from the task
    stream so they are not double-counted."""
