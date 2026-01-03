# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

from .physics_materials_cfg import StageSpecificRigidBodyMaterialCfg
from .visual_materials_cfg import StageSpecificPreviewSurfaceCfg

"""
PCV MATERIALS
"""


@configclass
class PCVVisualMaterialCfg(StageSpecificPreviewSurfaceCfg):
    diffuse_color = (0.5, 0.1, 0.1)
    roughness = 0.3
    metallic = 0.0
    opacity = 1.0


@configclass
class PCVPhysicalMaterialCfg(StageSpecificRigidBodyMaterialCfg):
    static_friction = 0.4
    dynamic_friction = 0.23
    restitution = 0.2


"""
STEEL MATERIALS
"""


@configclass
class SteelVisualMaterialCfg(StageSpecificPreviewSurfaceCfg):
    diffuse_color = (0.5, 0.5, 0.5)
    roughness = 0.3
    metallic = 0.9
    opacity = 1.0


@configclass
class SteelPhysicalMaterialCfg(StageSpecificRigidBodyMaterialCfg):
    static_friction = 0.445
    dynamic_friction = 0.375
    restitution = 0.56
