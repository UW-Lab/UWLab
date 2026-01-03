# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch


def aug_observation(obs: torch.Tensor) -> torch.Tensor:
    """
    obs: num_steps_per_env * num_envs // num_mini_batches, 449
    /////////////////////////////////////////
    0: [0:3] (3,) 'base_lin_vel'
    1: [3:6] (3,) 'base_ang_vel'
    2: [6:9] (3,) 'proj_gravity'
    3: [9:13] (3,) 'concatenate_cmd'
    4: [13:14] (1,) 'time_left'
    5: [14:26] (12,) 'joint_pos'
    6: [26:38] (12,) 'joint_vel'
    7: [38:50] (12,) 'last_actions'
    8: [50:] (400,) 'height_scan'
    ////////////////////////////////////////
    # (LF, LH, RF, RH) x (HAA, HFE, HKE)

    """
    B = obs.shape[0]
    new_obs = obs.repeat(4, 1)
    # Y-symmetry Front-Back Symmetry #
    # 0-2 base lin vel
    new_obs[B : 2 * B, 0] = -new_obs[B : 2 * B, 0]
    # 3-5 base ang vel
    new_obs[B : 2 * B, 4] = -new_obs[B : 2 * B, 4]
    new_obs[B : 2 * B, 5] = -new_obs[B : 2 * B, 5]
    # 6-8 proj gravity
    new_obs[B : 2 * B, 6] = -new_obs[B : 2 * B, 6]
    # 9-12 cmd
    new_obs[B : 2 * B, 9] = -new_obs[B : 2 * B, 9]
    new_obs[B : 2 * B, 12] = -new_obs[B : 2 * B, 12]
    # 13 time left(ignored)

    # origin -> y_symmetry
    # *F_HAA ->  *H_HAA
    # *F_HFE -> -*H_HFE
    # *F_KFE -> -*H_KFE
    # *H_HAA ->  *F_HAA
    # *H_HFE -> -*F_HFE
    # *H_KFE -> -*F_KFE

    # Original           vs       X-symmetry
    # 00 = 'LF_HAA'               01 = 'LH_HAA'
    # 01 = 'LH_HAA'               00 = 'LF_HAA'
    # 02 = 'RF_HAA'               03 = 'RH_HAA'
    # 03 = 'RH_HAA'               02 = 'RF_HAA'
    # 04 = 'LF_HFE'               05 = -'LH_HFE'
    # 05 = 'LH_HFE'               04 = -'LF_HFE'
    # 06 = 'RF_HFE'               07 = -'RH_HFE'
    # 07 = 'RH_HFE'               06 = -'RF_HFE'
    # 08 = 'LF_KFE'               09 = -'LH_KFE'
    # 09 = 'LH_KFE'               08 = -'LF_KFE'
    # 10 = 'RF_KFE'               11 = -'RH_KFE'
    # 11 = 'RH_KFE'               10 = -'RF_KFE'

    new_idx = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    dir_change = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1], device=obs.device)
    # left-right exchange state
    # 13-25 joint pos
    new_obs[B : 2 * B, 14:26] = new_obs[B : 2 * B, 14:26][:, new_idx] * dir_change
    # 25-37 joint vel
    new_obs[B : 2 * B, 26:38] = new_obs[B : 2 * B, 26:38][:, new_idx] * dir_change
    # 37-49 last_action
    new_obs[B : 2 * B, 38:50] = new_obs[B : 2 * B, 38:50][:, new_idx] * dir_change
    # height scan, x-y order,  (-length/2, -width/2) to (length/2, width/2)
    new_obs[B : 2 * B, 50:] = new_obs[B : 2 * B, 50:].reshape(B, 16, 25).flip(2).flatten(1, 2)

    # X-symmetry Left Right Symmetry#
    # 0-2 base lin vel
    new_obs[2 * B : 3 * B, 1] = -new_obs[2 * B : 3 * B, 1]
    # 3-5 base ang vel
    new_obs[2 * B : 3 * B, 3] = -new_obs[2 * B : 3 * B, 3]
    new_obs[2 * B : 3 * B, 5] = -new_obs[2 * B : 3 * B, 5]
    # 6-8 proj gravity
    new_obs[2 * B : 3 * B, 7] = -new_obs[2 * B : 3 * B, 7]
    # 9-12 cmd
    new_obs[2 * B : 3 * B, 10] = -new_obs[2 * B : 3 * B, 10]
    new_obs[2 * B : 3 * B, 12] = -new_obs[2 * B : 3 * B, 12]
    # 13 time left(ignored)
    # Original           vs       X-symmetry
    # 00 = 'LF_HAA'               02 = -'RF_HAA'
    # 01 = 'LH_HAA'               03 = -'RH_HAA'
    # 02 = 'RF_HAA'               00 = -'LF_HAA'
    # 03 = 'RH_HAA'               01 = -'LH_HAA'
    # 04 = 'LF_HFE'               06 = 'RF_HFE'
    # 05 = 'LH_HFE'               07 = 'RH_HFE'
    # 06 = 'RF_HFE'               04 = 'LF_HFE'
    # 07 = 'RH_HFE'               05 = 'LH_HFE'
    # 08 = 'LF_KFE'               10 = 'RF_KFE'
    # 09 = 'LH_KFE'               11 = 'RH_KFE'
    # 10 = 'RF_KFE'               08 = 'LF_KFE'
    # 11 = 'RH_KFE'               09 = 'LH_KFE'

    new_idx = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]
    dir_change = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1], device=obs.device)
    new_obs[2 * B : 3 * B, 14:26] = new_obs[2 * B : 3 * B, 14:26][:, new_idx] * dir_change
    new_obs[2 * B : 3 * B, 26:38] = new_obs[2 * B : 3 * B, 26:38][:, new_idx] * dir_change
    new_obs[2 * B : 3 * B, 38:50] = new_obs[2 * B : 3 * B, 38:50][:, new_idx] * dir_change
    new_obs[2 * B : 3 * B, 50:] = new_obs[2 * B : 3 * B, 50:].reshape(B, 16, 25).flip(1).flatten(1, 2)

    # X-Y-symmetry #
    # 0-2 base lin vel
    new_obs[3 * B : 4 * B, :2] = -new_obs[3 * B : 4 * B, :2]
    # 3-5 base ang vel
    new_obs[3 * B : 4 * B, 3] = -new_obs[3 * B : 4 * B, 3]
    new_obs[3 * B : 4 * B, 4] = -new_obs[3 * B : 4 * B, 4]
    # 6-8 proj gravity
    new_obs[3 * B : 4 * B, 6:8] = -new_obs[3 * B : 4 * B, 6:8]
    # 9-12 cmd
    new_obs[3 * B : 4 * B, 9:11] = -new_obs[3 * B : 4 * B, 9:11]
    # 13 time left(ignored)
    # Original           vs       XY-symmetry
    # 00 = 'LF_HAA'               03 = -'RH_HAA'
    # 01 = 'LH_HAA'               02 = -'RF_HAA'
    # 02 = 'RF_HAA'               01 = -'LH_HAA'
    # 03 = 'RH_HAA'               00 = -'LF_HAA'
    # 04 = 'LF_HFE'               07 = -'RH_HFE'
    # 05 = 'LH_HFE'               06 = -'RF_HFE'
    # 06 = 'RF_HFE'               05 = -'LH_HFE'
    # 07 = 'RH_HFE'               04 = -'LF_HFE'
    # 08 = 'LF_KFE'               11 = -'RH_KFE'
    # 09 = 'LH_KFE'               10 = -'RF_KFE'
    # 10 = 'RF_KFE'               09 = -'LH_KFE'
    # 11 = 'RH_KFE'               08 = -'LF_KFE'

    new_idx = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8]
    dir_change = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], device=obs.device)
    new_obs[3 * B : 4 * B, 14:26] = new_obs[3 * B : 4 * B, 14:26][:, new_idx] * dir_change
    new_obs[3 * B : 4 * B, 26:38] = new_obs[3 * B : 4 * B, 26:38][:, new_idx] * dir_change
    new_obs[3 * B : 4 * B, 38:50] = new_obs[3 * B : 4 * B, 38:50][:, new_idx] * dir_change
    new_obs[3 * B : 4 * B, 50:] = new_obs[3 * B : 4 * B, 50:].reshape(B, 16, 25).flip(1).flip(2).flatten(1, 2)

    return new_obs


def aug_actions(
    actions: torch.Tensor, actions_log_prob: torch.Tensor, action_mean: torch.Tensor, action_sigma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    new_actions = actions.repeat(4, 1)
    new_actions_log_prob = actions_log_prob.repeat(4, 1)
    new_action_mean = action_mean.repeat(4, 1)
    new_action_sigma = action_sigma.repeat(4, 1)
    B = actions.shape[0]
    # Y-symmetry Front-Back Symmetry #
    new_idx = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    dir_change = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1], device=actions.device)
    new_actions[B : 2 * B, :] = new_actions[B : 2 * B, new_idx] * dir_change
    new_action_mean[B : 2 * B, :] = new_action_mean[B : 2 * B, new_idx] * dir_change
    new_action_sigma[B : 2 * B, :] = new_action_sigma[B : 2 * B, new_idx]

    # X-symmetry Left Right Symmetry#
    new_idx = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]
    dir_change = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1], device=actions.device)
    new_actions[2 * B : 3 * B, :] = new_actions[2 * B : 3 * B, new_idx] * dir_change
    new_action_mean[2 * B : 3 * B, :] = new_action_mean[2 * B : 3 * B, new_idx] * dir_change
    new_action_sigma[2 * B : 3 * B, :] = new_action_sigma[2 * B : 3 * B, new_idx]

    # XY-symmetry
    new_idx = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8]
    dir_change = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], device=actions.device)
    new_actions[3 * B : 4 * B, :] = new_actions[3 * B : 4 * B, new_idx] * dir_change
    new_action_mean[3 * B : 4 * B, :] = new_action_mean[3 * B : 4 * B, new_idx] * dir_change
    new_action_sigma[3 * B : 4 * B, :] = new_action_sigma[3 * B : 4 * B, new_idx]

    return new_actions, new_actions_log_prob, new_action_mean, new_action_sigma


def aug_action(actions: torch.Tensor) -> torch.Tensor:
    new_actions = actions.repeat(4, 1)
    B = actions.shape[0]
    # Y-symmetry Front-Back Symmetry #
    new_idx = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
    dir_change = torch.tensor([1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1], device=actions.device)
    new_actions[B : 2 * B, :] = new_actions[B : 2 * B, new_idx] * dir_change

    # X-symmetry Left Right Symmetry#
    new_idx = [2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9]
    dir_change = torch.tensor([-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1], device=actions.device)
    new_actions[2 * B : 3 * B, :] = new_actions[2 * B : 3 * B, new_idx] * dir_change

    # XY-symmetry
    new_idx = [3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8]
    dir_change = torch.tensor([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], device=actions.device)
    new_actions[3 * B : 4 * B, :] = new_actions[3 * B : 4 * B, new_idx] * dir_change

    return new_actions


def aug_func(obs=None, actions=None, env=None):
    aug_obs = None
    aug_act = None
    if obs is not None:
        aug_obs = obs.repeat(4)
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
