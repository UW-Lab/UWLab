# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING

from isaaclab.utils import configclass

from . import patch_sampling as sampling_functions


@configclass
class PatchSamplingCfg:
    """Configuration for sampling patches on the sub-terrain."""

    func: Callable = MISSING
    """The function to use for sampling patches."""

    num_patches: int = MISSING
    """Number of patches to sample."""

    patch_radius: float | list[float] = MISSING
    """Radius of the patches."""

    patched = False

    def __post_init__(self):
        if not self.patched:
            cfg = self.to_dict()
            cfg["cfg"] = self.__class__
            setattr(self, "patch_radius", cfg)


@configclass
class FlatPatchSamplingCfg(PatchSamplingCfg):
    func: Callable = sampling_functions.find_flat_patches
    """The function to use for sampling patches."""

    patch_radius: float | list[float] = MISSING
    """Radius of the patches.

    A list of radii can be provided to check for patches of different sizes. This is useful to deal with
    cases where the terrain may have holes or obstacles in some areas.
    """

    x_range: tuple[float, float] = (-1e6, 1e6)
    """The range of x-coordinates to sample from. Defaults to (-1e6, 1e6).

    This range is internally clamped to the size of the terrain mesh.
    """

    y_range: tuple[float, float] = (-1e6, 1e6)
    """The range of y-coordinates to sample from. Defaults to (-1e6, 1e6).

    This range is internally clamped to the size of the terrain mesh.
    """

    z_range: tuple[float, float] = (-1e6, 1e6)
    """Allowed range of z-coordinates for the sampled patch. Defaults to (-1e6, 1e6)."""

    max_height_diff: float = MISSING
    """Maximum allowed height difference between the highest and lowest points on the patch."""


@configclass
class PieceWiseRangeFlatPatchSamplingCfg(PatchSamplingCfg):
    """Configuration for sampling flat patches on the sub-terrain with piece-wise ranges."""

    func: Callable = sampling_functions.find_piecewise_range_flat_patches
    """The function to use for sampling patches with piece wise ranges."""

    patch_radius: float | list[float] = MISSING
    """Radius of the patches.

    A list of radii can be provided to check for patches of different sizes. This is useful to deal with
    cases where the terrain may have holes or obstacles in some areas.
    """

    x_range: list[tuple[float, float]] | tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for X sampling (in mesh frame)."""

    y_range: list[tuple[float, float]] | tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for Y sampling (in mesh frame)."""

    z_range: list[tuple[float, float]] | tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for Z filtering (in mesh frame)."""

    max_height_diff: float = MISSING
    """Maximum allowed height difference between the highest and lowest points on the patch."""

    max_iterations: int = 100


@configclass
class FlatPatchSamplingByRadiusCfg(PatchSamplingCfg):
    func: Callable = sampling_functions.find_flat_patches_by_radius

    patch_radius: float | list[float] = MISSING

    radius_range: tuple[float, float] = MISSING

    x_range: list[tuple[float, float]] | tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for X sampling (in mesh frame)."""

    y_range: list[tuple[float, float]] | tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for Y sampling (in mesh frame)."""

    z_range: tuple[float, float] = (-1e6, 1e6)
    """The list of (min, max) intervals for Z filtering (in mesh frame)."""

    max_height_diff: float = MISSING

    max_iterations: int = 100
