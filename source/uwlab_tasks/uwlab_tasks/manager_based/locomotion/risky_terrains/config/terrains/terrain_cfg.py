# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.terrains.terrain_generator as terrain_generator
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from uwlab.terrains.trimesh import (
    MeshBalanceBeamsTerrainCfg,
    MeshSteppingBeamsTerrainCfg,
    MeshStonesEverywhereTerrainCfg,
)
from uwlab.terrains.utils import FlatPatchSamplingCfg, PatchSamplingCfg


def patched_find_flat_patches(*args, **kwargs) -> None:
    patch_key = "patch_radius"
    kwargs[patch_key]["patched"] = True
    cfg_class = kwargs[patch_key]["cfg"]
    cfg_class_args = {key: val for key, val in kwargs[patch_key].items() if key not in ["func", "cfg"]}
    patch_sampling_cfg: PatchSamplingCfg = cfg_class(**cfg_class_args)
    return patch_sampling_cfg.func(kwargs["wp_mesh"], kwargs["origin"], patch_sampling_cfg)


terrain_generator.find_flat_patches = patched_find_flat_patches


RISKY_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(10, 10),
    border_width=10,
    num_rows=10,
    num_cols=40,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stones_everywhere": MeshStonesEverywhereTerrainCfg(
            w_gap=(0.08, 0.26),
            w_stone=(0.92, 0.2),
            s_max=(0.036, 0.118),
            h_max=(0.01, 0.1),
            holes_depth=-10.0,
            platform_width=1.5,
        )
    },
)

BALANCE_BEAMS_CFG = TerrainGeneratorCfg(
    size=(10, 10),
    border_width=10,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "balance_beams": MeshBalanceBeamsTerrainCfg(
            platform_width=2.0,
            h_offset=(0.01, 0.1),
            w_stone=(0.25, 0.25),
            mid_gap=0.25,
            flat_patch_sampling={
                "target": FlatPatchSamplingCfg(
                    patch_radius=0.4,
                    num_patches=10,
                    x_range=(4, 6),
                    y_range=(-1, 1),
                    z_range=(-0.05, 0.05),
                    max_height_diff=0.05,
                )
            },
        ),
    },
)

STEPPING_BEAMS_CFG = TerrainGeneratorCfg(
    size=(10, 10),
    border_width=10,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stepping_beams": MeshSteppingBeamsTerrainCfg(
            platform_width=2.0,
            h_offset=(0.01, 0.1),
            w_stone=(0.5, 0.2),
            l_stone=(0.8, 1.6),
            gap=(0.15, 0.5),
            yaw=(0, 15),
        )
    },
)
