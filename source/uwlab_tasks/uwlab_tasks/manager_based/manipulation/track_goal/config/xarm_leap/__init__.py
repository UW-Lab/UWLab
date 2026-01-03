# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

"""
XarmLeap
"""
gym.register(
    id="UW-TrackGoal-XarmLeap-JointPos-Deployment-v0",
    entry_point="uwlab.envs.real_rl_env:RealRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_goal_xarm_leap:RealRLEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-TrackGoal-XarmLeap-JointPos-Viz-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_goal_xarm_leap:TrackGoalXarmLeapVizJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-TrackGoal-XarmLeap-JointPos-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_goal_xarm_leap:TrackGoalXarmLeapJointPosition",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="UW-TrackGoal-XarmLeap-IkAbs-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_goal_xarm_leap:TrackGoalXarmLeapMcIkAbs",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
        "teleop_cfg_entry_point": "uwlab_assets.robots.xarm_leap.teleop:XarmLeapTeleopCfg",
    },
    disable_env_checker=True,
)


gym.register(
    id="UW-TrackGoal-XarmLeap-IkDel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.track_goal_xarm_leap:TrackGoalXarmLeapMcIkDel",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:Base_PPORunnerCfg",
    },
    disable_env_checker=True,
)
