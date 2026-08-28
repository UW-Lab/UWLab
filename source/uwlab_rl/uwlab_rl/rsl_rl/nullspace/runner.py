# Copyright (c) 2026, Null-space preference critic project.
# SPDX-License-Identifier: BSD-3-Clause

"""On-policy runner that can construct the dual-critic policy and null-space PPO.

Upstream's ``_construct_algorithm`` resolves class names with a bare ``eval()`` evaluated in
*its own module namespace*, so classes defined outside ``rsl_rl.runners.on_policy_runner`` are
unreachable by name. This subclass overrides only that resolution step; everything else --
the rollout loop, logging, checkpointing -- is inherited unchanged.
"""

from __future__ import annotations

from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups  # noqa: F401  (kept for parity with upstream imports)

from .dual_actor_critic import DualCriticActorCritic
from .nullspace_ppo import NullspacePPO

#: Class names this runner can resolve, in addition to whatever upstream supports.
REGISTRY = {
    "DualCriticActorCritic": DualCriticActorCritic,
    "NullspacePPO": NullspacePPO,
}


class DualCriticOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner that resolves the null-space classes by name."""

    def _construct_algorithm(self, obs: TensorDict) -> NullspacePPO:
        policy_cfg = dict(self.policy_cfg)
        alg_cfg = dict(self.alg_cfg)

        policy_class_name = policy_cfg.pop("class_name", "DualCriticActorCritic")
        alg_class_name = alg_cfg.pop("class_name", "NullspacePPO")

        unknown = [n for n in (policy_class_name, alg_class_name) if n not in REGISTRY]
        if unknown:
            raise ValueError(
                f"{unknown} not resolvable by DualCriticOnPolicyRunner. Known: {sorted(REGISTRY)}. "
                "Use the stock OnPolicyRunner for upstream classes."
            )

        # RND is rejected by NullspacePPO; surface that here rather than deep in the algorithm.
        if alg_cfg.get("rnd_cfg") is not None:
            raise NotImplementedError("RND is not supported alongside the dual critic.")
        alg_cfg.pop("rnd_cfg", None)
        alg_cfg.pop("symmetry_cfg", None)

        policy = REGISTRY[policy_class_name](
            obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg
        ).to(self.device)

        alg = REGISTRY[alg_class_name](
            policy, device=self.device, **alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )

        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg
