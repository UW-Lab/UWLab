# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_risky(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    demotion_fraction: float = 0.1,
):
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("target_cmd")
    command_term = env.command_manager.get_term("target_cmd")

    goal_position_b = command[env_ids, :2]

    total_distance = (command_term.pos_command_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]).norm(2, dim=1)

    distance_to_goal = goal_position_b.norm(2, dim=1)
    move_up = distance_to_goal < 0.4
    move_down = (distance_to_goal / total_distance) > (1 - demotion_fraction)
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
