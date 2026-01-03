# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


def aug_observation(obs: torch.Tensor) -> torch.Tensor:
    """
    obs: num_steps_per_env * num_envs // num_mini_batches, 313
    /////////////////////////////////////////
    0: [0:3] (3,) 'base_lin_vel'
    1: [3:6] (3,) 'base_ang_vel'
    2: [6:9] (3,) 'proj_gravity'
    3: [9:13] (4,) 'concatenate_cmd'
    4: [13:14] (1,) 'time_left'
    5: [14:26] (12,) 'joint_pos'
    6: [26:38] (12,) 'joint_vel'
    7: [38:50] (12,) 'last_actions'
    8: [50:] (264,) 'height_scan'
    ////////////////////////////////////////

    """
    B = obs.shape[0]
    new_obs = obs.repeat(2, 1)

    # 0-2 base lin vel
    new_obs[B : 2 * B, 1] = -new_obs[B : 2 * B, 1]
    # 3-5 base ang vel
    new_obs[B : 2 * B, 3] = -new_obs[B : 2 * B, 3]
    new_obs[B : 2 * B, 5] = -new_obs[B : 2 * B, 5]
    # 6-8 proj gravity
    new_obs[B : 2 * B, 7] = -new_obs[B : 2 * B, 7]
    # 9-13 cmd
    new_obs[B : 2 * B, 10] = -new_obs[B : 2 * B, 10]
    new_obs[B : 2 * B, 12] = -new_obs[B : 2 * B, 12]

    # X-symmetry:
    # 00 = 'fl_hx' , 01 = fr_hx (-1)
    # 01 = 'fr_hx' , 00 = fl_hx (-1)
    # 02 = 'hl_hx' , 03 = hr_hx (-1)
    # 03 = 'hr_hx' , 02 = hl_hx (-1)
    # 04 = 'fl_hy' , 05 = fr_hy
    # 05 = 'fr_hy' , 04 = fl_hy
    # 06 = 'hl_hy' , 07 = hr_hy
    # 07 = 'hr_hy' , 06 = hl_hy
    # 08 = 'fl_kn' , 09 = fr_kn
    # 09 = 'fr_kn' , 08 = fl_kn
    # 10 = 'hl_kn' , 11 = hr_kn
    # 11 = 'hr_kn' , 10 = hl_kn

    new_idx = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    dir_change = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1], device=obs.device)
    new_obs[B : 2 * B, 14:26] = new_obs[B : 2 * B, 14:26][:, new_idx] * dir_change
    new_obs[B : 2 * B, 26:38] = new_obs[B : 2 * B, 26:38][:, new_idx] * dir_change
    new_obs[B : 2 * B, 38:50] = new_obs[B : 2 * B, 38:50][:, new_idx] * dir_change
    new_obs[B : 2 * B, 50:] = new_obs[B : 2 * B, 50:].reshape(B, 11, 24).flip(1).flatten(1, 2)

    return new_obs


def aug_actions(
    actions: torch.Tensor, actions_log_prob: torch.Tensor, action_mean: torch.Tensor, action_sigma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    new_actions = actions.repeat(2, 1)
    new_actions_log_prob = actions_log_prob.repeat(2, 1)
    new_action_mean = action_mean.repeat(2, 1)
    new_action_sigma = action_sigma.repeat(2, 1)
    B = actions.shape[0]

    new_idx = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    dir_change = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1], device=actions.device)
    new_actions[B : 2 * B, :] = new_actions[B : 2 * B, new_idx] * dir_change
    new_action_mean[B : 2 * B, :] = new_action_mean[B : 2 * B, new_idx] * dir_change
    new_action_sigma[B : 2 * B, :] = new_action_sigma[B : 2 * B, new_idx]

    return new_actions, new_actions_log_prob, new_action_mean, new_action_sigma


def aug_action(actions: torch.Tensor) -> torch.Tensor:
    new_actions = actions.repeat(2, 1)
    B = actions.shape[0]

    new_idx = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    dir_change = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1], device=actions.device)
    new_actions[B : 2 * B, :] = new_actions[B : 2 * B, new_idx] * dir_change

    return new_actions


def aug_func(obs=None, actions=None, env=None):
    aug_obs = None
    aug_act = None
    if obs is not None:
        aug_obs = obs.repeat(2)
        if "policy" in obs:
            aug_obs["policy"] = aug_observation(obs["policy"])
        elif "critic" in obs:
            aug_obs["critic"] = aug_observation(obs["critic"])
        else:
            raise ValueError(
                "nothing is augmented because not policy or critic keyword found in tensordict,                you"
                f" keys: {list(obs.keys())} \n please check for potential bug"
            )
    if actions is not None:
        aug_act = aug_action(actions)
    return aug_obs, aug_act
