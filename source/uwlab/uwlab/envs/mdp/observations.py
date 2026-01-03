# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs import ManagerBasedRLEnv


def life_spent(env: ManagerBasedRLEnv) -> torch.Tensor:
    if hasattr(env, "episode_length_buf"):
        life_spent = env.episode_length_buf.float() / env.max_episode_length
    else:
        life_spent = torch.zeros(env.num_envs, device=env.device, dtype=torch.float)
    return life_spent.view(-1, 1)
