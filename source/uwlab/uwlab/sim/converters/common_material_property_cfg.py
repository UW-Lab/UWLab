# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import field

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from ..spawners.materials import common_materials_cfg as common_materials
from .mesh_converter_cfg import MeshConverterCfg


@configclass
class PVCCfg(MeshConverterCfg):
    rigid_props: sim_utils.RigidBodyPropertiesCfg = field(default_factory=sim_utils.RigidBodyPropertiesCfg)
    collision_props: sim_utils.CollisionPropertiesCfg = field(default_factory=sim_utils.CollisionPropertiesCfg)
    mass_props: sim_utils.MassPropertiesCfg = sim_utils.MassPropertiesCfg(density=1380)
    visual_material_props: sim_utils.VisualMaterialCfg = field(default_factory=common_materials.PCVVisualMaterialCfg)
    physics_material_props: sim_utils.PhysicsMaterialCfg = field(
        default_factory=common_materials.PCVPhysicalMaterialCfg
    )


@configclass
class SteelCfg(MeshConverterCfg):
    rigid_props: sim_utils.RigidBodyPropertiesCfg = field(default_factory=sim_utils.RigidBodyPropertiesCfg)
    collision_props: sim_utils.CollisionPropertiesCfg = field(default_factory=sim_utils.CollisionPropertiesCfg)
    mass_props: sim_utils.MassPropertiesCfg = sim_utils.MassPropertiesCfg(density=7870)
    visual_material_props: sim_utils.VisualMaterialCfg = field(default_factory=common_materials.SteelVisualMaterialCfg)
    physics_material_props: sim_utils.PhysicsMaterialCfg = field(
        default_factory=common_materials.SteelPhysicalMaterialCfg
    )
