# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-package for environment definitions.

Environments define the interface between the agent and the simulation.
In the simplest case, the environment provides the agent with the current
observations and executes the actions provided by the agent. However, the
environment can also provide additional information such as the current
reward, done flag, and information about the current episode.

UW Lab provide Data-Manager-based and Real-Rl environment workflows:

* **Data-Manager-based**: The IsaacLab Manager Based Rl Environments is designed such that no manager role's is to
    hold and share fresh state data and complex data structure. The Command Manager can holds prev_step command
    that is used to condition on the policy but calculated data is inherently one step behind if used by
    Reward Manager or Event Managers. Data Manager calculates the state immediately after the action so the state
    information is always up to date. more, the data doesn't needs to be tensor and is designed to be shared across
    managers with data manager providing cfg for the data retrieval.

* **Real-Rl**: The Real-Rl Environments is designed to be used in real robot applications. The environment uses
    universal robot articulation to abstract the real hardware control loop and provides a simple interface to interact
    with the rest of uwlab and isaaclab tool kit. This provides seamless experience from simulation to real

Based on these workflows, there are the following environment classes for single and multi-agent RL:
"""

from .real_rl_env import RealRLEnv
from .real_rl_env_cfg import RealRLEnvCfg
