# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data manager for generating and updating data."""

from __future__ import annotations

import inspect
import torch
import weakref
from abc import abstractmethod
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import omni.kit.app
from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase

from .manager_term_cfg import DataTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DataTerm(ManagerTermBase):
    """The base class for implementing a data term.

    A data term encapsulates a specialized, up-to-date data structure intended for use by multiple managers.
    It is designed for scenarios where recalculating computationally expensive data repeatedly is inefficient.

    It is possible to assign a visualization function to the data term
    that can be used to visualize the data in the simulator.
    """

    def __init__(self, cfg: DataTermCfg, env: ManagerBasedRLEnv):
        """Initialize the data manager class.

        Args:
            cfg: The configuration parameters for the data manager.
            env: The environment object.
        """
        super().__init__(cfg, env)

        # create buffers to store the data
        # -- infos that can be used for logging
        self.infos = dict()
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        # set initial state of debug visualization
        self.set_debug_vis(self.cfg.debug_vis)

    def __del__(self):
        """Unsubscribe from the callbacks."""
        if self._debug_vis_handle:
            self._debug_vis_handle.unsubscribe()
            self._debug_vis_handle = None

    """
    Properties
    """

    @property
    @abstractmethod
    def data(self) -> torch.Tensor:
        """The data tensor. Shape is (num_envs, data_dim)."""
        raise NotImplementedError

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the data generator has a debug visualization implemented."""
        # check if function raises NotImplementedError
        source_code = inspect.getsource(self._set_debug_vis_impl)
        return "NotImplementedError" not in source_code

    """
    Operations.
    """

    def set_debug_vis(self, debug_vis: bool) -> bool:
        """Sets whether to visualize the data.

        Args:
            debug_vis: Whether to visualize the data.

        Returns:
            Whether the debug visualization was successfully set. False if the data
            generator does not support debug visualization.
        """
        # check if debug visualization is supported
        if not self.has_debug_vis_implementation:
            return False
        # toggle debug visualization objects
        self._set_debug_vis_impl(debug_vis)
        # toggle debug visualization handles
        if debug_vis:
            # create a subscriber for the post update event if it doesn't exist
            if self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None
        # return success
        return True

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset the data manager and log infos.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "{name}" key.
        """
        # resolve the environment IDs
        if env_ids is None:
            env_ids = slice(None)

        # add logging infos
        extras = {}
        for info_name, info_value in self.infos.items():
            # compute the mean info value
            extras[info_name] = torch.mean(info_value[env_ids]).item()
            # reset the info value
            info_value[env_ids] = 0.0

        return extras

    def compute(self, dt: float):
        """Compute the data.

        Args:
            dt: The time step passed since the last call to compute.
        """
        # update the infos based on current state
        self._update_infos()
        # update the data
        self._update_data()

    """
    Implementation specific functions.
    """

    @abstractmethod
    def _update_infos(self):
        """Update the infos based on the current state."""
        raise NotImplementedError

    @abstractmethod
    def _update_data(self):
        """Update the data based on the current state."""
        raise NotImplementedError

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Set debug visualization into visualization objects.

        This function is responsible for creating the visualization objects if they don't exist
        and input ``debug_vis`` is True. If the visualization objects exist, the function should
        set their visibility into the stage.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")

    def _debug_vis_callback(self, event):
        """Callback for debug visualization.

        This function calls the visualization objects and sets the data to visualize into them.
        """
        raise NotImplementedError(f"Debug visualization is not implemented for {self.__class__.__name__}.")


class DataManager(ManagerBase):
    """Manager for generating data.

    DataManager is designed to address the challenge of managing computationally expensive
    data structures that must be shared across multiple system components. In many scenarios,
    these data structures—such as reorganized indices of multi-dimensional tensors—are not easily
    derived directly from the base asset data, and recalculating them repeatedly would be inefficient.

    - "Expensive" refers to data structures that require significant computational resources to construct.
    Specifically, these are reorganized indices of multi-dimensional tensors that all managers rely on and
    cannot be directly derived from the base Isaac Lab asset data without a costly computation process.

    - "Up-to-date" means that the data is current and fresh, reflecting the most recent state of the system
    without delay. For example, if the reward and event managers were to use data computed by the
    command manager—which performs its computations later—they would end up using information that is
    already one step behind, showing that command manager is not a appropriate place to calculate fresh data.
    Other managers are not appropriate either because their names and concepts are unrelated.

    Centralizing the data generation within the DataManager ensures that all dependent managers receive
    synchronized and immediately relevant data, maintaining optimal system performance.

    The data terms are implemented as classes that inherit from the :class:`DataTerm` class.
    Each data manager term should also have a corresponding configuration class that inherits from the
    :class:`DataTermCfg` class.
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initialize the data manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, DataTermCfg]``).
            env: The environment instance.
        """
        # create buffers to parse and store terms
        self._terms: dict[str, DataTerm] = dict()

        # call the base class constructor (this prepares the terms)
        super().__init__(cfg, env)
        # store the data
        self._data = dict()
        if self.cfg:
            self.cfg.debug_vis = False
            for term in self._terms.values():
                self.cfg.debug_vis |= term.cfg.debug_vis

    def __str__(self) -> str:
        """Returns: A string representation for the data manager."""
        msg = f"<DataManager> contains {len(self._terms.values())} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Data Terms"
        table.field_names = ["Index", "Name", "Type"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, (name, term) in enumerate(self._terms.items()):
            table.add_row([index, name, term.__class__.__name__])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active data terms."""
        return list(self._terms.keys())

    @property
    def has_debug_vis_implementation(self) -> bool:
        """Whether the data terms have debug visualization implemented."""
        # check if function raises NotImplementedError
        has_debug_vis = False
        for term in self._terms.values():
            has_debug_vis |= term.has_debug_vis_implementation
        return has_debug_vis

    """
    Operations.
    """

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """

        terms = []
        idx = 0
        for name, term in self._terms.items():
            terms.append((name, term.data[env_idx].cpu().tolist()))
            idx += term.data.shape[1]
        return terms

    def set_debug_vis(self, debug_vis: bool):
        """Sets whether to visualize the data data.

        Args:
            debug_vis: Whether to visualize the data data.

        Returns:
            Whether the debug visualization was successfully set. False if the data
            generator does not support debug visualization.
        """
        for term in self._terms.values():
            term.set_debug_vis(debug_vis)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Reset the data terms and log their infos.

        Args:
            env_ids: The list of environment IDs to reset. Defaults to None.

        Returns:
            A dictionary containing the information to log under the "Infos/{term_name}/{info_name}" key.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # store information
        extras = {}
        for name, term in self._terms.items():
            # reset the data term
            infos = term.reset(env_ids=env_ids)
            # compute the mean info value
            for info_name, info_value in infos.items():
                extras[f"Infos/{name}/{info_name}"] = info_value
        # return logged information
        return extras

    def compute(self, dt: float):
        """Updates the data.

        This function calls each data term managed by the class.

        Args:
            dt: The time-step interval of the environment.

        """
        # iterate over all the data terms
        for term in self._terms.values():
            # compute term's value
            term.compute(dt)

    def get_data(self, name: str) -> torch.Tensor:
        """Returns the data for the specified data term.

        Args:
            name: The name of the data term.

        Returns:
            The data tensor of the specified data term.
        """
        return self._terms[name].data

    def get_term(self, name: str) -> DataTerm:
        """Returns the data term with the specified name.

        Args:
            name: The name of the data term.

        Returns:
            The data term with the specified name.
        """
        return self._terms[name]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, DataTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type DataTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # create the action term
            term = term_cfg.class_type(term_cfg, self._env)
            # sanity check if term is valid type
            if not isinstance(term, DataTerm):
                raise TypeError(f"Returned object for the term '{term_name}' is not of type DataType.")
            # add class to dict
            self._terms[term_name] = term
