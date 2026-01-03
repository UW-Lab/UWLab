# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

gym.register(
    id="UW-Position-Advance-Skills-Arm-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:AdvanceSkillsSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AdvanceSkillsSpotPPORunnerCfg",
    },
)

gym.register(
    id="UW-Position-Pit-Arm-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:PitSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AdvanceSkillsSpotPPORunnerCfg",
    },
)

gym.register(
    id="UW-Position-Gap-Arm-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:GapSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AdvanceSkillsSpotPPORunnerCfg",
    },
)

gym.register(
    id="UW-Position-Inv-Slope-Arm-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:SlopeInvSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AdvanceSkillsSpotPPORunnerCfg",
    },
)

gym.register(
    id="UW-Position-Extreme-Stair-Arm-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:ExtremeStairSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AdvanceSkillsSpotPPORunnerCfg",
    },
)

gym.register(
    id="UW-Position-Square-Obstacle-Arm-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:SquarePillarObstacleSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AdvanceSkillsSpotPPORunnerCfg",
    },
)

gym.register(
    id="UW-Position-Irregular-Obstacle-Arm-Spot-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.spot_env_cfg:IrregularPillarObstacleSpotEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AdvanceSkillsSpotPPORunnerCfg",
    },
)
