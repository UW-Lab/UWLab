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


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    demotion_fraction: float = 0.05,
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to reach a desired location.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than specified distance required by the position command.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("goal_point")
    # compute the distance the robot walked
    distance = command[env_ids, :2].norm(2, dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance < 0.5
    # robots that walked less than half of their required distance go to simpler terrains
    total_distance = (
        env.command_manager.get_term("goal_point").pos_command_w[env_ids, :2] - env.scene.env_origins[env_ids, :2]
    ).norm(2, dim=1)
    distance_traveled = 1 - distance / total_distance
    move_down = distance_traveled < demotion_fraction
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())
