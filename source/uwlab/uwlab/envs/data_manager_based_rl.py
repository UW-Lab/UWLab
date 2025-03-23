# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedEnvCfg, ManagerBasedRLEnv
from isaaclab.envs.manager_based_env import VecEnvObs
from isaaclab.managers import EventManager
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.timer import Timer
from uwlab.scene import InteractiveScene

from ..managers.data_manager import DataManager
from .data_manager_based_rl_cfg import DataManagerBasedRLEnvCfg

VecEnvStepReturn = tuple[VecEnvObs, torch.Tensor, torch.Tensor, torch.Tensor, dict]


class DataManagerBasedRLEnv(ManagerBasedRLEnv):
    """The superclass for flexible reinforcement learning-based environments.

    This class extending the :class:`ManagerBasedRLEnv` and implements the flexible modules
    that makes experimentation and hacking easier while still using manager based style.
    Specifically, the two fields, extensions and data manager.
    The field self.extension is a dictionary serves as a place to store any data that needs
    a centralized access that can not be modularized by managers, this is usually the last
    resort when no other optional are easy and it is not recommended to use in any formalized code.
    Data Manager in another hand, can be used to provide more structured data structure to be accessed
    by rest of managers. In practice, this module should be used carefully and avoid at best
    when formalizing the code and only use for advanced special treatment.

    """

    cfg: DataManagerBasedRLEnvCfg
    """Configuration for the environment."""

    def __init__(self, cfg: DataManagerBasedRLEnvCfg, render_mode: str | None = None, **kwargs):
        self.extensions = {}
        # there is no good way to override the InteractiveScene class, so we need to patch it
        self._interactive_scene_patched_manager_based_env_init(cfg)
        self._manager_based_rl_env_init(cfg, render_mode)
        # RLTaskEnv.__bases__ = (BaseEnv, gym.Env)
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.02, 0.02, 0.02)
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        self.symmetry_augmentation_func = None
        if hasattr(cfg, "symmetry_augmentation_func"):
            self.symmetry_augmentation_func = cfg.symmetry_augmentation_func

    def _interactive_scene_patched_manager_based_env_init(self, cfg: ManagerBasedEnvCfg):
        import builtins

        import omni.log

        from isaaclab.envs.ui import ViewportCameraController
        from isaaclab.sim import SimulationContext

        # check that the config is valid
        cfg.validate()
        # store inputs to class
        self.cfg = cfg
        # initialize internal variables
        self._is_closed = False

        # set the seed for the environment
        if self.cfg.seed is not None:
            self.cfg.seed = self.seed(self.cfg.seed)
        else:
            omni.log.warn("Seed not set for the environment. The environment creation may not be deterministic.")

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            # the type-annotation is required to avoid a type-checking error
            # since it gets confused with Isaac Sim's SimulationContext class
            self.sim: SimulationContext = SimulationContext(self.cfg.sim)
        else:
            # simulation context should only be created before the environment
            # when in extension mode
            if not builtins.ISAAC_LAUNCHED_FROM_TERMINAL:
                raise RuntimeError("Simulation context already exists. Cannot create a new one.")
            self.sim: SimulationContext = SimulationContext.instance()

        # make sure torch is running on the correct device
        if "cuda" in self.device:
            torch.cuda.set_device(self.device)

        # print useful information
        print("[INFO]: Base environment:")
        print(f"\tEnvironment device    : {self.device}")
        print(f"\tEnvironment seed      : {self.cfg.seed}")
        print(f"\tPhysics step-size     : {self.physics_dt}")
        print(f"\tRendering step-size   : {self.physics_dt * self.cfg.sim.render_interval}")
        print(f"\tEnvironment step-size : {self.step_dt}")

        if self.cfg.sim.render_interval < self.cfg.decimation:
            msg = (
                f"The render interval ({self.cfg.sim.render_interval}) is smaller than the decimation "
                f"({self.cfg.decimation}). Multiple render calls will happen for each environment step. "
                "If this is not intended, set the render interval to be equal to the decimation."
            )
            omni.log.warn(msg)

        # counter for simulation steps
        self._sim_step_counter = 0

        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(self.cfg.scene)
        print("[INFO]: Scene manager: ", self.scene)

        # set up camera viewport controller
        # viewport is not available in other rendering modes so the function will throw a warning
        # FIXME: This needs to be fixed in the future when we unify the UI functionalities even for
        # non-rendering modes.
        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller = ViewportCameraController(self, self.cfg.viewer)
        else:
            self.viewport_camera_controller = None

        # create event manager
        # note: this is needed here (rather than after simulation play) to allow USD-related randomization events
        #   that must happen before the simulation starts. Example: randomizing mesh scale
        self.event_manager = EventManager(self.cfg.events, self)
        print("[INFO] Event Manager: ", self.event_manager)

        # apply USD-related randomization events
        if "prestartup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="prestartup")

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
            print("[INFO]: Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()
            # add timeline event to load managers
            self.load_managers()

        # extend UI elements
        # we need to do this here after all the managers are initialized
        # this is because they dictate the sensors and commands right now
        if self.sim.has_gui() and self.cfg.ui_window_class_type is not None:
            # setup live visualizers
            self.setup_manager_visualizers()
            self._window = self.cfg.ui_window_class_type(self, window_name="IsaacLab")
        else:
            # if no window, then we don't need to store the window
            self._window = None

        # allocate dictionary to store metrics
        self.extras = {}

        # initialize observation buffers
        self.obs_buf = {}

    def _manager_based_rl_env_init(self, cfg: DataManagerBasedRLEnvCfg, render_mode):
        # store the render mode
        self.render_mode = render_mode

        # initialize data and constants
        # -- counter for curriculum
        self.common_step_counter = 0
        # -- init buffers
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        # -- set the framerate of the gym video recorder wrapper so that the playback speed of the produced video matches the simulation
        self.metadata["render_fps"] = 1 / self.step_dt

        print("[INFO]: Completed setting up the environment...")

    @property
    def is_closed(self):
        return self._is_closed

    """
    Patches
    """

    def patch_interactive_scene(self):
        import sys

        # Make sure we have not imported the original module yet
        if "isaaclab.scene" in sys.modules:
            del sys.modules["isaaclab.scene"]

        import uwlab.scene

        sys.modules["isaaclab.scene"] = uwlab.scene

    def unpatch_interactive_scene(self):
        import sys

        if "isaaclab.scene" in sys.modules:
            del sys.modules["isaaclab.scene"]

    """
    Operations - MDP
    """

    def step(self, action: torch.Tensor) -> VecEnvStepReturn:
        """Execute one time-step of the environment's dynamics and reset terminated environments.

        Unlike the :class:`ManagerBasedEnvCfg.step` class, the function performs the following operations:

        1. Process the actions.
        2. Perform physics stepping.
        3. Perform rendering if gui is enabled.
        4. Update the environment counters and compute the rewards and terminations.
        5. Reset the environments that terminated.
        6. Compute the observations.
        7. Return the observations, rewards, resets and extras.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        # process actions
        self.action_manager.process_action(action)

        self.recorder_manager.record_pre_step()

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            # set actions into buffers
            self.action_manager.apply_action()
            # set actions into simulator
            self.scene.write_data_to_sim()
            # simulate
            self.sim.step(render=False)
            # render between steps only if the GUI or an RTX sensor needs it
            # note: we assume the render interval to be the shortest accepted rendering interval.
            #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render()
            # update buffers at sim dt
            self.scene.update(dt=self.physics_dt)

        if hasattr(self, "data_manager"):
            self.data_manager.compute()
        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)
        # -- check terminations
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        # -- reward computation
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
            self.scene.write_data_to_sim()
            self.sim.forward()

            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.rerender_on_reset:
                self.sim.render()

            # trigger recorder terms for post-reset calls
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

    def load_managers(self):
        if getattr(self.cfg, "data", None) is not None:
            self.data_manager: DataManager = DataManager(self.cfg.data, self)
            print("[INFO] Data Manager: ", self.data_manager)
        super().load_managers()

    def _reset_idx(self, env_ids: Sequence[int]):
        if getattr(self, "data_manager", None) is not None:
            self.data_manager.reset(env_ids)
        return super()._reset_idx(env_ids)

    def close(self):
        for key, val in self.extensions.items():
            del val
        super().close()
