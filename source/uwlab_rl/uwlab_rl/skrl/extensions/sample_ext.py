# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from skrl.agents.torch.base import Agent

"""
Sample extension returns a function that takes a batch of samples
and return the additional samples wish to be added to the memory.
"""


def experts_act_f(
    env: ManagerBasedRLEnv,
    agent: Agent,
    context: dict,
    map_encoding_to_expert_key: str,
):
    """
    Creates an action function for agents using terrain-specific expert policies based on encoding-based mapping.

    Parameters:
    ----------
    env : ManagerBasedRLEnv
        The IsaacLab Manager-based RL environment

    agent : Agent
        The SKRL Agent

    context : dict
        A dictionary containing various configuration and context-specific information, including expert policies.

    map_encoding_to_expert_key : str
        The key used to access `expert_encoding_policies_dict` from `context`.

    Returns:
    -------
    Callable[[dict], torch.Tensor]
        An action function `act` that takes a batch of states and returns terrain-based actions specific to each agent.

    Detailed Functionality:
    -----------------------
    - Retrieves expert policies using `map_encoding_to_expert_key` from `context`.
    - Maps environment terrain encodings to expert policies, aligning each agent with a policy based on their
      current terrain.
    - The `act` function generates actions by processing a batch of states and returning terrain-specific actions
      for each agent according to `agent_terrain_id` from the environment.

    Example Usage:
    --------------
    ```
    action_function = experts_act_f(env, agent, context, "terrain_expert_policies")
    actions = action_function({'states': torch.rand(env.num_envs, observation_dim)})
    ```

    Notes:
    ------
    - The function utilizes terrain encodings from the environment to match agents with appropriate expert policies
      from `expert_encoding_policies_dict`.
    - The `act` function recalculates agent-appropriate actions based on the `agent_terrain_id` at each step,
      allowing dynamic policy application if agent terrain changes over time.
    - The `act` function returns a clone of the generated action tensor to ensure isolation from in-place modifications.
    """

    expert_encoding_policies_dict: dict[tuple, Callable[[torch.Tensor], torch.Tensor]] = context[
        map_encoding_to_expert_key
    ]

    # Order of .keys() and .values() is guaranteed to be the same in Python 3.7+
    expert_encodings = list(expert_encoding_policies_dict.keys())
    expert_policies = list(expert_encoding_policies_dict.values())
    expert_encodings = torch.tensor(expert_encodings, device=env.device)
    terrain_encodings_tensor = env.extensions["terrain_encoding_cache"]
    order = [torch.where((expert_encodings == t).all(dim=1))[0][0] for t in terrain_encodings_tensor]
    ordered_expert_policies = [expert_policies[i] for i in order]

    act_dim = env.action_space.shape[0]  # type: ignore
    actions = torch.zeros((env.num_envs, act_dim)).to(env.device)

    def act(batch):
        observations = batch["states"]
        # query agent_terrain_id in the loop to consider the case where agent changes terrain
        agent_terrain_id = env.scene.terrain.agent_terrain_id  # type: ignore
        for i in range(len(ordered_expert_policies)):
            actions[agent_terrain_id == i] = ordered_expert_policies[i](observations[agent_terrain_id == i])
        return actions.clone()

    return act
