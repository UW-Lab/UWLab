# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sim.converters.asset_converter_base_cfg import AssetConverterBaseCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.sim.spawners.materials import PhysicsMaterialCfg
from isaaclab.sim.spawners.materials.visual_materials_cfg import VisualMaterialCfg
from isaaclab.utils import configclass


@configclass
class MeshConverterCfg(AssetConverterBaseCfg):
    """The configuration class for MeshConverter."""

    visual_material_props: VisualMaterialCfg | None = None
    """Visual material properties to apply to the USD. Defaults to None.

    Note:
        If None, then default visual properties will be added.
    """

    physics_material_props: PhysicsMaterialCfg | None = None
    """Physics material properties to apply to the USD. Defaults to None.

    Note:
        If None, then default physics material properties will be added.
    """

    mass_props: schemas_cfg.MassPropertiesCfg | None = None
    """Mass properties to apply to the USD. Defaults to None.

    Note:
        If None, then no mass properties will be added.
    """

    rigid_props: schemas_cfg.RigidBodyPropertiesCfg | None = None
    """Rigid body properties to apply to the USD. Defaults to None.

    Note:
        If None, then no rigid body properties will be added.
    """

    collision_props: schemas_cfg.CollisionPropertiesCfg | None = None
    """Collision properties to apply to the USD. Defaults to None.

    Note:
        If None, then no collision properties will be added.
    """

    collision_approximation: str = "convexDecomposition"
    """Collision approximation method to use. Defaults to "convexDecomposition".

    Valid options are:
    "convexDecomposition", "convexHull", "boundingCube",
    "boundingSphere", "meshSimplification", or "none"

    "none" causes no collision mesh to be added.
    """
