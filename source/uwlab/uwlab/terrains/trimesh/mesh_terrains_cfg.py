# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
from typing import Literal

from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass

import uwlab.terrains.trimesh.mesh_terrains as mesh_terrains

"""
Different trimesh terrain configurations.
"""


@configclass
class MeshObjTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.obj_terrain

    obj_path: str = MISSING

    spawn_origin_path: str = MISSING


@configclass
class CachedTerrainGenCfg(MeshObjTerrainCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.cached_terrain_gen

    height: float = MISSING

    levels: float = MISSING

    include_overhang: bool = MISSING

    task_descriptor: str = MISSING


@configclass
class TerrainGenCfg(MeshObjTerrainCfg):
    """Configuration for a plane mesh terrain."""

    function = mesh_terrains.terrain_gen

    height: float = MISSING

    levels: float = MISSING

    include_overhang: bool = MISSING

    terrain_styles: list = MISSING

    yaml_path: str = (MISSING,)

    spawn_origin_path: str = MISSING

    python_script: str = MISSING


@configclass
class MeshStonesEverywhereTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with stones everywhere
    """

    function = mesh_terrains.stones_everywhere_terrain

    # stone gap width
    w_gap: tuple[float, float] = MISSING

    # grid square stone size (width)
    w_stone: tuple[float, float] = MISSING

    # the maximum shift, both x and y shift is uniformly sample from [-s_max, s_max]
    s_max: tuple[float, float] = MISSING

    # the maximum height, the height is uniformly sample from [-hmax, h_max], default height is 1.0 m
    h_max: tuple[float, float] = MISSING

    # holes depth
    holes_depth: float = MISSING

    # the platform width
    platform_width: float = MISSING


@configclass
class MeshBalanceBeamsTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with balance-beams
    """

    # balance beams terrain function
    function = mesh_terrains.balance_beams_terrain

    # the platform width
    platform_width: float = MISSING

    # the height offset
    h_offset: tuple[float, float] = MISSING

    # stone width
    w_stone: tuple[float, float] = MISSING

    # the gap between two beams
    mid_gap: float = MISSING


@configclass
class MeshSteppingBeamsTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with stepping-beams
    """

    # stepping beams terrain function
    function = mesh_terrains.stepping_beams_terrain

    # the platform width
    platform_width: float = MISSING

    # the height offset
    h_offset: tuple[float, float] = MISSING

    # stone width
    w_stone: tuple[float, float] = MISSING

    # length of the stepping beams
    l_stone: tuple[float, float] = MISSING

    #  the gap between two beams
    gap: tuple[float, float] = MISSING

    # the yaw angle of the stepping beams
    yaw: tuple[float, float] = MISSING


@configclass
class MeshDiversityBoxTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with boxes for anymal parkour
    """

    function = mesh_terrains.box_terrain

    # the box width range
    box_width_range: tuple[float, float] = MISSING
    # the box length range
    box_length_range: tuple[float, float] = MISSING
    # the box height range
    box_height_range: tuple[float, float] = MISSING

    # the gap between two boxes
    box_gap_range: tuple[float, float] = None  # type: ignore

    # flag for climbing up (box is set at the origin ) or climb down (box is set near the origin)
    up_or_down: str = None  # type: ignore


@configclass
class MeshPassageTerrainCfg(SubTerrainBaseCfg):
    """
    A terrain with passage
    """

    function = mesh_terrains.passage_terrain

    # the passage width (y dir)
    passage_width: float | tuple[float, float] = MISSING

    # the passage height
    passage_height: float | tuple[float, float] = MISSING

    # the passage length (x dir)
    passage_length: float | tuple[float, float] = MISSING


@configclass
class MeshStructuredTerrainCfg(SubTerrainBaseCfg):
    """Configuration for a structured terrain."""

    function = mesh_terrains.structured_terrain
    terrain_type: Literal["stairs", "inverted_stairs", "obstacles", "walls"] = MISSING
