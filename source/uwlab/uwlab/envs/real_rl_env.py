# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import isaacsim.core.utils.torch as torch_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import VecEnvObs, VecEnvStepReturn
from isaaclab.managers import (
    ActionManager,
    CommandManager,
    CurriculumManager,
    EventManager,
    ObservationManager,
    RecorderManager,
    RewardManager,
    TerminationManager,
)
from isaaclab.utils.timer import Timer

if TYPE_CHECKING:
    from .real_rl_env_cfg import RealRLEnvCfg


class RealRLEnv(ManagerBasedRLEnv):
    """EXPERIMENTAL FEATURE
    The class is an experiment feature that uses IsaacLab API to directly control real robot
    through the usage of Universal Articulation Assets. The benefit is that one can control the
    real robot with exactly the managers and existing terms in IsaacLab, reducing the needs to
    switch to a different workflow. The RealRL Env follows the implementation of ManagerBasedRLEnv
    but ignores or remove the simulation specific parameters.
    """

    is_vector_env: ClassVar[bool] = True
    """Whether the environment is a vectorized environment."""

    metadata: ClassVar[dict[str, Any]] = {}
    """real environment doesn't have metadata"""

    cfg: RealRLEnvCfg

    def __init__(self, cfg: RealRLEnvCfg, **kwargs) -> None:
        # cfg.validate() # TODO: uncomment this line when bug resolved
        self.cfg = cfg
        self.cfg.scene.device = self.cfg.device
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            print("Seed not set for the environment. The environment creation may not be deterministic.")

        with Timer("[INFO]: Time taken for context scene creation", "scene_creation"):
            self.scene = self.cfg.scene.class_type(self.cfg.scene)
        print("[INFO]: Scene manager: ", self.scene)

        # -- counter for sanity check
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
        with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
            self.scene.start()
        self.load_managers()

        self._sim_step_counter = 0

        # allocate dictionary to store metrics
        self.extras = {}

        # initialize observation buffers
        self.obs_buf = {}

    """
    Operations - Setup.
    """

    @property
    def num_envs(self) -> int:
        """The number of instances of the environment that are running."""
        return 1

    @property
    def device(self):
        """The device on which the environment is running."""
        return self.cfg.device

    @property
    def max_episode_length_s(self) -> float:
        """Maximum episode length in seconds."""
        return self.cfg.episode_length_s

    @property
    def max_episode_length(self) -> int:
        """Maximum episode length in environment steps."""
        return math.ceil(self.max_episode_length_s / self.step_dt)

    @property
    def physic_dt(self) -> float:
        raise NotImplementedError(f"The property physic_dt is not supported for {self.__class__.__name__}.")

    @property
    def step_dt(self) -> float:
        return self.cfg.scene.dt

    """
    Operations - Setup.
    """

    def load_managers(self):
        # note: this order is important since observation manager needs to know the command and action managers
        # and the reward manager needs to know the termination manager
        # -- command manager
        self.command_manager: CommandManager = CommandManager(self.cfg.commands, self)
        print("[INFO] Command Manager: ", self.command_manager)

        self.recorder_manager = RecorderManager(self.cfg.recorders, self)
        print("[INFO] Recorder Manager: ", self.recorder_manager)
        # -- action manager
        self.action_manager = ActionManager(self.cfg.actions, self)
        print("[INFO] Action Manager: ", self.action_manager)
        # -- observation manager
        self.observation_manager = ObservationManager(self.cfg.observations, self)
        print("[INFO] Observation Manager:", self.observation_manager)
        # -- event manager
        self.event_manager = EventManager(self.cfg.events, self)
        print("[INFO] Event Manager: ", self.event_manager)

        # -- termination manager
        self.termination_manager = TerminationManager(self.cfg.terminations, self)
        print("[INFO] Termination Manager: ", self.termination_manager)
        # # -- reward manager
        self.reward_manager = RewardManager(self.cfg.rewards, self)
        print("[INFO] Reward Manager: ", self.reward_manager)
        # -- curriculum manager
        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        print("[INFO] Curriculum Manager: ", self.curriculum_manager)

        # setup the action and observation spaces for Gym
        self._configure_gym_env_spaces()

        # perform events at the start of the simulation
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        self.action_manager.process_action(action.to(self.device))

        self.recorder_manager.record_pre_step()

        # perform physics stepping
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_context()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # # -- reward computation
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            # update observations for recording if needed
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # trigger recorder terms for pre-reset calls
            self.recorder_manager.record_pre_reset(reset_env_ids)

            self._reset_idx(reset_env_ids)
            # update articulation kinematics
            self.scene.write_data_to_context()

            # # trigger recorder terms for post-reset calls
            self.recorder_manager.record_post_reset(reset_env_ids)

        # -- update command
        self.command_manager.compute(dt=self.step_dt)
        # -- step interval events
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        # -- compute observations
        # note: done after reset to get the correct observations for reset envs
        self.obs_buf = self.observation_manager.compute()

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def render(self, recompute: bool = False) -> np.ndarray | None:
        raise NotImplementedError(f"The function render is not supported for {self.__class__.__name__}.")

    def reset(
        self, seed: int | None = None, env_ids: Sequence[int] | None = None, options: dict[str, Any] | None = None
    ) -> tuple[VecEnvObs, dict]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, dtype=torch.int64, device=self.device)

        # set the seed
        if seed is not None:
            self.seed(seed)

        # reset state of scene
        self._reset_idx(env_ids)

        # update articulation kinematics
        self.scene.write_data_to_context()

        # compute observations
        self.obs_buf = self.observation_manager.compute()

        self.extras = dict()

        return self.obs_buf, self.extras

    def seed(self, seed: int = -1) -> int:
        return torch_utils.set_seed(seed)

    def close(self) -> None:
        del self.command_manager
        del self.reward_manager
        del self.termination_manager
        del self.action_manager
        del self.observation_manager
        del self.event_manager

    def _configure_gym_env_spaces(self):
        super()._configure_gym_env_spaces()

    def _reset_idx(self, env_ids: Sequence[int]):
        """Reset environments based on specified indices.

        Args:
            env_ids: List of environment ids which must be reset
        """
        # update the curriculum for environments that need a reset
        self.curriculum_manager.compute(env_ids=env_ids)
        # reset the internal buffers of the scene elements
        self.scene.reset(env_ids)
        # apply events such as randomizations for environments that need a reset
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(mode="reset", env_ids=env_ids, global_env_step_count=self.common_step_counter)

        # iterate over all managers and reset them
        # this returns a dictionary of information which is stored in the extras
        # note: This is order-sensitive! Certain things need be reset before others.
        self.extras["log"] = dict()
        # -- observation manager
        info = self.observation_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- action manager
        info = self.action_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- rewards manager
        info = self.reward_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- curriculum manager
        info = self.curriculum_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- command manager
        info = self.command_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- event manager
        info = self.event_manager.reset(env_ids)
        self.extras["log"].update(info)
        # -- termination manager
        info = self.termination_manager.reset(env_ids)
        self.extras["log"].update(info)

        # reset the episode length buffer
        self.episode_length_buf[env_ids] = 0
