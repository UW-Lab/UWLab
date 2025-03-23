# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from contextlib import contextmanager

from rsl_rl.algorithms.ppo import RolloutStorage


class AugmentiveGeneratorPatch:
    def __init__(self, symmetry_func):
        self.original_mini_batch_generator = RolloutStorage.mini_batch_generator
        self.original_reccurent_mini_batch_generator = RolloutStorage.recurrent_mini_batch_generator
        self.obs_aug_fn = symmetry_func[0]
        self.critic_obs_aug_fn = symmetry_func[1]
        self.act_aug_fn = symmetry_func[2]

    def apply_patch(self):
        original_mini_batch_generator = self.original_mini_batch_generator
        obs_aug_fn = self.obs_aug_fn
        critic_aug_fn = self.critic_obs_aug_fn
        act_aug_fn = self.act_aug_fn

        def augmented_mini_batch_generator(self, num_mini_batches, num_epochs=8):
            # Call the original generator and iterate over its yields
            for (
                obs_batch,
                critic_observations_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                extra,
                extra2,
                rnd_state_batch,
            ) in original_mini_batch_generator(self, num_mini_batches, num_epochs):

                # Apply augmentations
                old_dim = obs_batch.shape[0]
                obs_batch = obs_aug_fn(obs_batch)
                new_dim = obs_batch.shape[0]
                assert new_dim % old_dim == 0
                aug_ratio = new_dim // old_dim

                critic_observations_batch = critic_aug_fn(critic_observations_batch)
                actions_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch = act_aug_fn(
                    actions_batch, old_actions_log_prob_batch, old_mu_batch, old_sigma_batch
                )

                # Example: repeat target_values, returns, and advantages
                target_values_batch = target_values_batch.repeat(aug_ratio, 1)
                returns_batch = returns_batch.repeat(aug_ratio, 1)
                advantages_batch = advantages_batch.repeat(aug_ratio, 1)

                # Yield the augmented batch
                yield (
                    obs_batch,
                    critic_observations_batch,
                    actions_batch,
                    target_values_batch,
                    advantages_batch,
                    returns_batch,
                    old_actions_log_prob_batch,
                    old_mu_batch,
                    old_sigma_batch,
                    extra,
                    extra2,
                    rnd_state_batch,
                )

        RolloutStorage.mini_batch_generator = augmented_mini_batch_generator

    def remove_patch(self):
        RolloutStorage.mini_batch_generator = self.original_mini_batch_generator
        RolloutStorage.recurrent_mini_batch_generator = self.original_reccurent_mini_batch_generator


@contextmanager
def patch_augmentive_generator(symmetry_func):
    patcher = AugmentiveGeneratorPatch(symmetry_func)
    patcher.apply_patch()
    try:
        yield
    finally:
        patcher.remove_patch()
