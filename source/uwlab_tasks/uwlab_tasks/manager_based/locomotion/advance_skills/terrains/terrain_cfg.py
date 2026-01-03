# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains.terrain_generator as terrain_generator
from isaaclab import terrains as terrain_cfg

from uwlab.terrains.utils import FlatPatchSamplingByRadiusCfg, PatchSamplingCfg


def patched_find_flat_patches(*args, **kwargs) -> None:
    patch_key = "patch_radius"
    kwargs[patch_key]["patched"] = True
    cfg_class = kwargs[patch_key]["cfg"]
    cfg_class_args = {key: val for key, val in kwargs[patch_key].items() if key not in ["func", "cfg"]}
    patch_sampling_cfg: PatchSamplingCfg = cfg_class(**cfg_class_args)
    return patch_sampling_cfg.func(kwargs["wp_mesh"], kwargs["origin"], patch_sampling_cfg)


terrain_generator.find_flat_patches = patched_find_flat_patches


GAP = terrain_cfg.MeshGapTerrainCfg(
    platform_width=3.0,
    gap_width_range=(0.05, 1.5),
    proportion=0.2,
    flat_patch_sampling={
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            radius_range=(2.0, 4.0),
            max_height_diff=0.2,
        )
    },
)

PIT = terrain_cfg.MeshPitTerrainCfg(
    platform_width=3.0,
    pit_depth_range=(0.05, 1.2),
    proportion=0.2,
    flat_patch_sampling={
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(2.0, 4.0),
        )
    },
)

SQUARE_PILLAR_OBSTACLE = terrain_cfg.HfDiscreteObstaclesTerrainCfg(
    num_obstacles=8,
    obstacle_height_mode="fixed",
    obstacle_height_range=(2.0, 3.0),
    obstacle_width_range=(0.5, 1.5),
    proportion=0.2,
    platform_width=2,
    flat_patch_sampling={
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(4.0, 6.0),
        )
    },
)

IRREGULAR_PILLAR_OBSTACLE = terrain_cfg.MeshRepeatedBoxesTerrainCfg(
    platform_width=2,
    max_height_noise=0.5,
    proportion=0.2,
    object_params_start=terrain_cfg.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=5, height=4.0, size=(0.5, 0.5), max_yx_angle=0.0, degrees=True
    ),
    object_params_end=terrain_cfg.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
        num_objects=10, height=6.0, size=(1.0, 1.0), max_yx_angle=0.0, degrees=True
    ),
    flat_patch_sampling={
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(4.0, 6.0),
        )
    },
)

SLOPE_INV = terrain_cfg.HfInvertedPyramidSlopedTerrainCfg(
    proportion=0.2,
    slope_range=(0.0, 0.9),
    platform_width=2.0,
    border_width=1.5,
    flat_patch_sampling={
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(3.0, 4.5),
        )
    },
)

EXTREME_STAIR = terrain_cfg.HfPyramidStairsTerrainCfg(
    platform_width=3.0,
    step_height_range=(0.05, 0.2),
    step_width=0.3,
    proportion=0.2,
    inverted=True,
    border_width=1.0,
    flat_patch_sampling={
        "target": FlatPatchSamplingByRadiusCfg(
            num_patches=10,
            patch_radius=0.5,
            max_height_diff=0.2,
            radius_range=(4.0, 4.5),
        )
    },
)
