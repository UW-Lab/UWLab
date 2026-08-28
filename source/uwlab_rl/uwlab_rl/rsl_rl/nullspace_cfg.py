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
    ``separate``: an independent preference critic MLP (no coupling, so the beta=0 overlay is
    guaranteed by construction). Run sanity check A against both."""


@configclass
class RslRlNullspacePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO whose preference gradient is projected into the null space of the task gradient."""

    class_name: str = "NullspacePPO"

    beta: float = 0.0
    """Preference step budget. 0.0 reproduces baseline PPO exactly (the second backward pass is
    skipped), which is what makes sanity run A a true no-op."""

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


@configclass
class RslRlNullspaceRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner that can resolve the null-space classes by name."""

    class_name: str = "DualCriticOnPolicyRunner"

    pref_source: str = "zero"
    """Preference reward stream: ``zero`` (sanity run A), ``noise`` (sanity run B, an
    uninformative preference that the projection must neutralise), or ``terms`` (Phase 2 scripted
    manner predicates read out of the RewardManager)."""

    pref_noise_std: float = 1.0
    """Std of the Gaussian preference reward when ``pref_source='noise'``."""

    pref_term_names: tuple[str, ...] = ()
    """RewardManager term names forming the preference stream when ``pref_source='terms'``.
    These stay in the manager (evaluation order untouched) but are subtracted from the task
    stream so they are not double-counted."""
