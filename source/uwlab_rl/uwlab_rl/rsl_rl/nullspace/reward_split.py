# Copyright (c) 2026, Null-space preference critic project.
# SPDX-License-Identifier: BSD-3-Clause

"""Split the environment's scalar reward into a task stream and a preference stream.

Design constraint discovered in Phase 0 recon: OmniReset's ``RewardsCfg`` contains
``progress_context``, a term that returns zeros but caches the goal distances that ``r_dist``,
``r_success``, three termination terms, the reset-distribution success monitor and the parent
project's data-collection configs all read back off it. ``RewardManager`` also *skips any term
whose weight is 0.0* without calling it. So splitting the manager into two managers, reordering
terms, or zeroing weights are all ways to silently corrupt the reward.

We therefore never touch the RewardManager. It already materialises per-term rewards in
``_step_reward`` of shape (num_envs, num_terms); we read that buffer and mask it.

Note on scaling: ``reward_manager.compute`` sets ``_step_reward[:, i] = func_i * weight_i`` and
adds ``func_i * weight_i * dt`` to the reward buffer. So ``_step_reward`` has the weight *already
applied* and ``dt`` divided out -- the contribution of term i to the returned reward is
``_step_reward[:, i] * dt``, with no second weight multiply.
"""

from __future__ import annotations

import torch
from abc import ABC, abstractmethod

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper


class PreferenceRewardSource(ABC):
    """Where the preference reward stream comes from.

    Phase 1 uses Zero (sanity run A) and GaussianNoise (sanity run B); Phase 2 uses
    RewardManagerTerms with scripted manner predicates; Phase 3 swaps in a distilled critic.
    """

    #: True when this source draws from terms that are *already inside* the RewardManager, and
    #: whose contribution must therefore be subtracted out of the task stream to avoid
    #: double-counting. False for exogenous sources (noise, a learned model).
    subtracts_from_task: bool = False

    def initialize(self, env) -> None:  # noqa: ANN001
        """Optional hook once the env exists (resolve term names to indices, etc.)."""

    @abstractmethod
    def compute(self, env, total_reward: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        """Return the per-env preference reward for this step, shape (num_envs,)."""


class ZeroPreference(PreferenceRewardSource):
    """No preference signal. Sanity run A: β=0 must reproduce the baseline."""

    def compute(self, env, total_reward: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        return torch.zeros_like(total_reward)


class GaussianNoisePreference(PreferenceRewardSource):
    """Zero-mean Gaussian preference reward. Sanity run B.

    Tests that the projection *neutralises an uninformative preference* rather than merely
    injecting variance: with β=1 the curves must still overlay the baseline.
    """

    def __init__(self, std: float = 1.0, seed: int | None = None) -> None:
        self.std = std
        self.seed = seed
        self._gen: torch.Generator | None = None

    def initialize(self, env) -> None:  # noqa: ANN001
        if self.seed is not None:
            self._gen = torch.Generator(device=env.unwrapped.device).manual_seed(self.seed)

    def compute(self, env, total_reward: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        return torch.normal(
            mean=0.0,
            std=self.std,
            size=total_reward.shape,
            device=total_reward.device,
            generator=self._gen,
        )


class ActionRatePreference(PreferenceRewardSource):
    """The **noise-bait probe**: preference = negative action-rate norm.

    This is literally ``r_smooth``'s second term (``action_rate_l2_clamped``), chosen because it is
    *maximally* satisfiable by shrinking exploration noise and barely satisfiable any other way.

    It exists to test the scoping of the null-space constraint, and it is strictly sharper than the
    Gaussian sanity run: zero-mean noise produces an *unsystematic* preference gradient that will
    not preferentially shrink exploration, so that run passes even with the leak wide open.

    Expected result when correctly scoped: **almost no compliance gain**, because both routes are
    closed -- the noise parameters are masked out of the preference gradient, and the mean policy
    is held by the task constraint. Large apparent compliance means the leak is still open;
    escalate to Layer 2 (``pref_detach_noise_features``).

    Exogenous (not subtracted from the task stream): the task's own ``action_rate`` term stays
    where it is, so the arms remain comparable to the baseline on task reward.
    """

    def compute(self, env, total_reward: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        am = env.unwrapped.action_manager
        return -torch.clamp(torch.sum(torch.square(am.action - am.prev_action), dim=1), 0, 1e4)


class RewardManagerTermsPreference(PreferenceRewardSource):
    """Preference reward = the sum of named RewardManager terms (Phase 2 scripted predicates).

    The named terms stay in the manager (so evaluation order and the ``progress_context``
    dependency chain are untouched) but their contribution is moved out of the task stream.
    """

    subtracts_from_task = True

    def __init__(self, term_names: list[str]) -> None:
        self.term_names = list(term_names)
        self._idx: torch.Tensor | None = None

    def initialize(self, env) -> None:  # noqa: ANN001
        rm = env.unwrapped.reward_manager
        known = list(rm.active_terms)
        missing = [n for n in self.term_names if n not in known]
        if missing:
            raise ValueError(
                f"Preference terms {missing} are not active RewardManager terms. Active: {known}"
            )
        self._idx = torch.tensor(
            [known.index(n) for n in self.term_names], dtype=torch.long, device=env.unwrapped.device
        )

    def compute(self, env, total_reward: torch.Tensor) -> torch.Tensor:  # noqa: ANN001
        rm = env.unwrapped.reward_manager
        # _step_reward already has per-term weights applied and dt divided out.
        return rm._step_reward[:, self._idx].sum(dim=-1) * env.unwrapped.step_dt


class DualRewardVecEnvWrapper(RslRlVecEnvWrapper):
    """Emits the task reward as the usual scalar and the preference reward via ``extras``.

    The preference stream rides in ``extras["reward_pref"]`` so that the rest of the rsl_rl
    plumbing (runner loop signature, env interface) is unchanged; only our PPO subclass reads it.
    """

    PREF_KEY = "reward_pref"

    def __init__(self, env, pref_source: PreferenceRewardSource | None = None, **kwargs) -> None:  # noqa: ANN001
        super().__init__(env, **kwargs)
        self.pref_source = pref_source or ZeroPreference()
        self.pref_source.initialize(self)

    def step(self, actions: torch.Tensor):  # noqa: ANN201
        obs, rew, dones, extras = super().step(actions)

        r_pref = self.pref_source.compute(self, rew)
        r_task = rew - r_pref if self.pref_source.subtracts_from_task else rew

        extras[self.PREF_KEY] = r_pref
        return obs, r_task, dones, extras
