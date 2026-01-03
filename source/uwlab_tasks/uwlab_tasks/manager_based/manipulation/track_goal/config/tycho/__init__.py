# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="UW-Track-Goal-Tycho-IkDel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tycho_track_goal:GoalTrackingTychoIkdelta",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-Track-Goal-Tycho-IkAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tycho_track_goal:GoalTrackingTychoIkabsolute",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
        "teleop_cfg_entry_point": "uwlab_assets.robots.tycho.teleop:TychoTeleopCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-Track-Goal-Tycho-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.tycho_track_goal:GoalTrackingTychoJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)
