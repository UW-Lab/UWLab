# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from isaaclab.sensors import SensorBase, SensorBaseCfg

from uwlab.assets import ArticulationCfg, UniversalArticulation

if TYPE_CHECKING:
    from .scene_context_cfg import SceneContextCfg


class SceneContext:
    """EXPERIMENTAL FEATURE
    Different from :class`InteractiveScene`, SceneContext are scene designed to allow more asset
    type, not limited to IsaacSim to be added in SceneContext, this is useful for RealEnv to incorporate
    real hard ware in synchronize with the Isaac Lab simulation loop.
    """

    def __init__(self, cfg: SceneContextCfg):
        """Initializes the scene.

        Args:
            cfg: The configuration class for the scene.
        """
        # check that the config is valid
        cfg.validate()  # type: ignore
        # store inputs
        self.cfg = cfg
        # initialize scene elements
        self._terrain = None
        self._articulations = dict()
        self._deformable_objects = dict()
        self._rigid_objects = dict()
        self._rigid_object_collections = dict()
        self._sensors = dict()
        self._extras = dict()

        # physics scene path
        self._physics_scene_path = None

        self._global_prim_paths = list()

        if self._is_scene_setup_from_cfg():
            # add entities from config
            self._add_entities_from_cfg()
            # clone environments on a global scope if environment is homogeneous

    def __str__(self) -> str:
        """Returns a string representation of the scene."""
        msg = f"<class {self.__class__.__name__}>\n"
        msg += f"\tNumber of environments: {self.cfg.num_envs}\n"
        msg += f"\tGlobal prim paths     : {self._global_prim_paths}\n"
        return msg

    """
    Properties.
    """

    @property
    def physics_dt(self) -> float:
        """The physics timestep of the scene."""
        return self.cfg.dt

    @property
    def device(self) -> str:
        """The device on which the scene is created."""
        return self.cfg.device

    @property
    def num_envs(self) -> int:
        """The number of environments handled by the scene."""
        return self.cfg.num_envs

    @property
    def articulations(self) -> dict[str, UniversalArticulation]:
        """A dictionary of articulations in the scene."""
        return self._articulations

    @property
    def deformable_objects(self) -> dict[str, any]:
        """A dictionary of deformable objects in the scene."""
        return self._deformable_objects

    @property
    def rigid_objects(self) -> dict[str, any]:
        """A dictionary of rigid objects in the scene."""
        return self._rigid_objects

    @property
    def rigid_object_collections(self) -> dict[str, any]:
        """A dictionary of rigid object collections in the scene."""
        return self._rigid_object_collections

    @property
    def sensors(self) -> dict[str, SensorBase]:
        """A dictionary of the sensors in the scene, such as cameras and contact reporters."""
        return self._sensors

    @property
    def extras(self) -> dict[str, Any]:
        """A dictionary of extra entities in the scene, such as XFormPrimView."""
        return self._extras

    @property
    def env_origins(self) -> torch.Tensor:
        return torch.zeros((self.num_envs, 3), device=self.device)

    """
    Operations.
    """

    def start(self):
        """Starts the scene entities."""
        # -- assets
        for articulation in self._articulations.values():
            articulation._initialize_impl(self.device)
        # -- sensors
        for sensor in self._sensors.values():
            sensor._initialize_impl(self.device)

    def reset(self, env_ids: Sequence[int] | None = None):
        """Resets the scene entities.

        Args:
            env_ids: The indices of the environments to reset.
                Defaults to None (all instances).
        """
        # -- assets
        for articulation in self._articulations.values():
            articulation.reset(env_ids)
        # -- sensors
        for sensor in self._sensors.values():
            sensor.reset(env_ids)

    def write_data_to_context(self):
        """Writes the data of the scene entities to the simulation."""
        # -- assets
        for articulation in self._articulations.values():
            articulation.write_data_to_sim()

    def update(self, dt: float) -> None:
        """Update the scene entities.

        Args:
            dt: The amount of time passed from last :meth:`update` call.
        """
        # -- assets
        for articulation in self._articulations.values():
            articulation.update(dt)
        # -- sensors
        for sensor in self._sensors.values():
            sensor.update(dt, force_recompute=not self.cfg.lazy_sensor_update)

    """
    Operations: Iteration.
    """

    def keys(self) -> list[str]:
        """Returns the keys of the scene entities.

        Returns:
            The keys of the scene entities.
        """
        all_keys = []
        for asset_family in [
            self._articulations,
            self._sensors,
            self._extras,
        ]:
            all_keys += list(asset_family.keys())
        return all_keys

    def __getitem__(self, key: str) -> Any:
        """Returns the scene entity with the given key.

        Args:
            key: The key of the scene entity.

        Returns:
            The scene entity.
        """

        all_keys = []
        # check if it is in other dictionaries
        for asset_family in [
            self._articulations,
            self._sensors,
            self._extras,
        ]:
            out = asset_family.get(key)
            # if found, return
            if out is not None:
                return out
            all_keys += list(asset_family.keys())
        # if not found, raise error
        raise KeyError(f"Scene entity with key '{key}' not found. Available Entities: '{all_keys}'")

    """
    Internal methods.
    """

    def _is_scene_setup_from_cfg(self):
        from .scene_context_cfg import SceneContextCfg

        return any(
            not (asset_name in SceneContextCfg.__dataclass_fields__ or asset_cfg is None)
            for asset_name, asset_cfg in self.cfg.__dict__.items()
        )

    def _add_entities_from_cfg(self):
        """Add scene entities from the config."""
        # store paths that are in global collision filter
        self._global_prim_paths = list()
        from .scene_context_cfg import SceneContextCfg

        # parse the entire scene config and resolve regex
        for asset_name, asset_cfg in self.cfg.__dict__.items():
            # skip keywords
            # note: easier than writing a list of keywords: [num_envs, env_spacing, lazy_sensor_update]
            if asset_name in SceneContextCfg.__dataclass_fields__ or asset_cfg is None:
                continue
            # create asset
            if isinstance(asset_cfg, ArticulationCfg):
                self._articulations[asset_name] = asset_cfg.class_type(asset_cfg)
            elif isinstance(asset_cfg, SensorBaseCfg):
                self._sensors[asset_name] = asset_cfg.class_type(asset_cfg)
            else:
                raise ValueError(f"Unknown asset config type for {asset_name}: {asset_cfg}")
            # store global collision paths
