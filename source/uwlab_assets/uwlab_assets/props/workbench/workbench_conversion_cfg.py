# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from uwlab.sim.converters import MeshConverterCfg
from uwlab.sim.converters.common_material_property_cfg import PVCCfg, SteelCfg
from uwlab.sim.spawners.materials import common_materials_cfg as common_materials

BLOCK = PVCCfg(
    asset_path="datasets/workbench/block.stl",
    usd_dir="datasets/workbench",
    usd_file_name="block.usda",
    collision_approximation="convexDecomposition",
    force_usd_conversion=True,
    visual_material_props=common_materials.PCVVisualMaterialCfg(diffuse_color=(0.1, 0.8, 0.1)),
)

BOX = PVCCfg(
    asset_path="datasets/workbench/box.stl",
    usd_dir="datasets/workbench",
    usd_file_name="box.usda",
    collision_approximation="convexDecomposition",
    force_usd_conversion=True,
)

SHELF = SteelCfg(
    asset_path="datasets/workbench/shelf.stl",
    usd_dir="datasets/workbench",
    usd_file_name="shelf.usda",
    collision_approximation="convexDecomposition",
    force_usd_conversion=True,
    visual_material_props=common_materials.SteelVisualMaterialCfg(diffuse_color=(0.1, 0.1, 0.4)),
)

WORKBENCH_CONVERSION_CFG: list[MeshConverterCfg] = [BLOCK, BOX, SHELF]
