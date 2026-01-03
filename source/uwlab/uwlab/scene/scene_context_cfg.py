# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from collections.abc import Callable

from isaaclab.utils.configclass import configclass

from .scene_context import SceneContext


@configclass
class SceneContextCfg:
    class_type: Callable[..., SceneContext] = SceneContext

    dt: float = 0.02
    """Time step of the simulation. Default is 0.02."""

    device: str = "cpu"
    """Device to run the simulation on. Default is "cpu"."""

    num_envs: int = 1
    """Number of environment instances handled by the scene."""
    # DO NOT MODIFY NUM_ENVS DEFAULT VALUE, REAL ENVIRONMENT IS UNLIKELY TO HAVE MORE THAN 1 ENVIRONMENT

    lazy_sensor_update: bool = True
    """Whether to update sensors only when they are accessed. Default is True.

    If true, the sensor data is only updated when their attribute ``data`` is accessed. Otherwise, the sensor
    data is updated every time sensors are updated.
    """
