# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

from isaaclab.envs import ManagerBasedRLEnv


def time_left(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "extensions"):
        setattr(env, "extensions", {})
    if "max_episode_length_s" in env.extensions:
        if hasattr(env, "episode_length_buf"):
            life_left = (1 - (env.episode_length_buf.float() / env.extensions["max_episode_length"])) * env.extensions[
                "max_episode_length_s"
            ]
        else:
            life_left = (
                torch.ones(env.num_envs, device=env.device, dtype=torch.float) * env.extensions["max_episode_length_s"]
            )
    else:
        life_left = torch.ones(env.num_envs, device=env.device, dtype=torch.float) * env.max_episode_length_s
    return life_left.view(-1, 1)


def generated_modified_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """The generated command from command term in the command manager with the given name."""
    return torch.cat(
        (env.command_manager.get_command(command_name)[:, :2], env.command_manager.get_command(command_name)[:, 3:4]),
        dim=-1,
    )
