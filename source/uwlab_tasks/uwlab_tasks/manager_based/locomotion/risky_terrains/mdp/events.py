# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_episode_length_s(
    env: ManagerBasedRLEnv, envs_id: torch.Tensor, episode_length_s: tuple[float, float] = (5.0, 7.0)
):
    if "max_episode_length" not in env.extensions.keys():
        env.extensions["max_episode_length_s"] = (
            torch.rand(env.num_envs, device=env.device) * (episode_length_s[1] - episode_length_s[0])
            + episode_length_s[0]
        )
        env.extensions["max_episode_length"] = (env.extensions["max_episode_length_s"] / env.step_dt).to(torch.int16)

    else:
        env.extensions["max_episode_length_s"][envs_id] = (
            torch.rand(envs_id.shape[0], device=env.device) * (episode_length_s[1] - episode_length_s[0])
            + episode_length_s[0]
        )
        env.extensions["max_episode_length"][envs_id] = (
            env.extensions["max_episode_length_s"][envs_id] / env.step_dt
        ).to(torch.int16)
