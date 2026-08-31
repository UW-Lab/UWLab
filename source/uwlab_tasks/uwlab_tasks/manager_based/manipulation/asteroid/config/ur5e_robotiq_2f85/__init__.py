# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Reset-state recording environments
##

gym.register(
    id="Asteroid-UR5eRobotiq2f85-ObjectAnywhereEEAnywhere-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:PickObjectAnywhereEEAnywhereResetStatesCfg"},
)

gym.register(
    id="Asteroid-UR5eRobotiq2f85-ObjectRestingEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:PickObjectRestingEEGraspedResetStatesCfg"},
)

gym.register(
    id="Asteroid-UR5eRobotiq2f85-ObjectAnywhereEEGrasped-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": f"{__name__}.reset_states_cfg:PickObjectAnywhereEEGraspedResetStatesCfg"},
)

##
# State-based RL (expert) environments
##

gym.register(
    id="Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-State-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85PickRelCartesianOSCTrainCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AsteroidPPORunnerCfg",
    },
)

gym.register(
    id="Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85PickRelCartesianOSCFinetuneCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AsteroidPPORunnerCfg",
    },
)

gym.register(
    id="Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-State-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85PickRelCartesianOSCEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AsteroidPPORunnerCfg",
    },
)

gym.register(
    id="Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-State-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rl_state_cfg:Ur5eRobotiq2f85PickRelCartesianOSCFinetuneEvalCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AsteroidPPORunnerCfg",
    },
)

##
# Tactile (proprioceptive student) data collection / evaluation environments
##

gym.register(
    id="Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_tactile_cfg:Ur5eRobotiq2f85DataCollectionTactileRelCartesianOSCCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AsteroidDAggerRunnerCfg",
    },
)

gym.register(
    id="Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Finetune-DataCollection-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_tactile_cfg:Ur5eRobotiq2f85DataCollectionFinetuneTactileRelCartesianOSCCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AsteroidDAggerRunnerCfg",
    },
)

gym.register(
    id="Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.data_collection_tactile_cfg:Ur5eRobotiq2f85EvalTactileRelCartesianOSCCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AsteroidDAggerRunnerCfg",
    },
)

gym.register(
    id="Asteroid-Ur5eRobotiq2f85-RelCartesianOSC-Tactile-Finetune-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.data_collection_tactile_cfg:Ur5eRobotiq2f85EvalFinetuneTactileRelCartesianOSCCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cfg:AsteroidDAggerRunnerCfg",
    },
)
