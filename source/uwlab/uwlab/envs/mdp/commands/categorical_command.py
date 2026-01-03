# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import CUBOID_MARKER_CFG

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import CategoricalCommandCfg


class CategoricalCommand(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from uniform distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    If the :attr:`cfg.heading_command` flag is set to True, the angular velocity is computed from the heading
    error similar to doing a proportional control on the heading error. The target heading is sampled uniformly
    from the provided range. Otherwise, the angular velocity is sampled uniformly from the provided range.

    Mathematically, the angular velocity is computed as follows from the heading command:

    .. math::

        \omega_z = \frac{1}{2} \text{wrap_to_pi}(\theta_{\text{target}} - \theta_{\text{current}})

    """

    cfg: CategoricalCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: CategoricalCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        # initialize the base class
        super().__init__(cfg, env)  # type: ignore

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        self.category = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.num_categories = self.cfg.num_category

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tNumber of command categories: {self.cfg.num_category}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.category

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update the metrics of the command generator."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # category
        self.category[env_ids] = torch.randint(0, self.num_categories, (len(env_ids),), device=self.device)

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "category_visualizer"):
                # -- goal
                marker_cfg = CUBOID_MARKER_CFG.copy()  # type: ignore
                marker_cfg.prim_path = "/Visuals/Command/category"
                marker_cfg.markers["cuboid"].scale = (0.1, 0.1, 0.1)
                marker_cfg.markers["cuboid"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
                self.category_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.category_visualizer.set_visibility(True)
        else:
            if hasattr(self, "category_visualizer"):
                self.category_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_quat_w = self.robot.data.root_quat_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales
        scale = self.command[:].repeat_interleave(3, 0).view(-1, 3)

        self.category_visualizer.visualize(base_pos_w, base_quat_w, scale)
