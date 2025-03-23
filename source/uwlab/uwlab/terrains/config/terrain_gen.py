# Copyright (c) 2024-2025, The UW Lab Project Developers.
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict

import isaaclab.terrains as terrain_gen
import uwlab.terrains as uw_terrain_gen
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg

TERRAIN_GEN_SUB_TERRAINS: Dict[str, SubTerrainBaseCfg] = {
    "perlin": uw_terrain_gen.CachedTerrainGenCfg(
        proportion=1.00,
        height=0.25,
        levels=3,
        task_descriptor="perlin",
        include_overhang=False,
    ),
    "ramp_perlin": uw_terrain_gen.CachedTerrainGenCfg(
        proportion=1.00,
        height=0.50,
        levels=3,
        task_descriptor="ramp_perlin",
        include_overhang=False,
    ),
    "wall": uw_terrain_gen.CachedTerrainGenCfg(
        proportion=1.00,
        height=0.25,
        levels=3,
        task_descriptor="wall",
        include_overhang=False,
    ),
    "stair_ramp": uw_terrain_gen.CachedTerrainGenCfg(
        proportion=1.00,
        height=0.25,
        levels=3,
        task_descriptor="stair_ramp",
        include_overhang=False,
    ),
    "stair_platform": uw_terrain_gen.CachedTerrainGenCfg(
        proportion=1.00,
        height=0.25,
        levels=3,
        task_descriptor="stair_platform",
        include_overhang=False,
    ),
    "ramp": uw_terrain_gen.CachedTerrainGenCfg(
        proportion=1.00,
        height=0.25,
        levels=3,
        task_descriptor="ramp",
        include_overhang=False,
    ),
    "stair_platform_wall": uw_terrain_gen.CachedTerrainGenCfg(
        proportion=1.00,
        height=0.25,
        levels=3,
        task_descriptor="stair_platform_wall",
        include_overhang=False,
    ),
    "perlin_wall": uw_terrain_gen.CachedTerrainGenCfg(
        proportion=1.00,
        height=0.25,
        levels=3,
        task_descriptor="perlin_wall",
        include_overhang=False,
    ),
    "box": uw_terrain_gen.CachedTerrainGenCfg(
        proportion=1.00,
        height=0.25,
        levels=3,
        task_descriptor="box",
        include_overhang=False,
    ),
    "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        proportion=1.00,
        step_height_range=(0.05, 0.07),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
        holes=False,
    ),
    "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=1.00,
        step_height_range=(0.05, 0.07),
        step_width=0.3,
        platform_width=3.0,
        border_width=1.0,
        holes=False,
    ),
    "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=1.00, grid_width=0.45, grid_height_range=(0.45, 0.57), platform_width=2.0
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=1.00, noise_range=(0.02, 0.04), noise_step=0.02, border_width=0.25
    ),
    "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
        proportion=1.00, slope_range=(0.02, 0.04), platform_width=2.0, border_width=0.25
    ),
    "random_grid": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=1.00,
        platform_width=1.5,
        grid_width=0.75,
        grid_height_range=(0.025, 0.045),
        holes=False,
    ),
    "discrete_obstacle": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        proportion=1.00,
        size=(8.0, 8.0),
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_width=0.0,
        num_obstacles=100,
        obstacle_height_mode="choice",
        obstacle_width_range=(0.25, 0.75),
        obstacle_height_range=(1.0, 2.0),
        platform_width=1.5,
    ),
}
