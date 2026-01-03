# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
from isaaclab.sim.utils import clone, safe_set_attribute_on_usd_schema
from pxr import PhysxSchema, Usd, UsdPhysics, UsdShade

if TYPE_CHECKING:
    from . import physics_materials_cfg


@clone
def stage_specific_spawn_rigid_body_material(
    prim_path: str,
    cfg: physics_materials_cfg.StageSpecificRigidBodyMaterialCfg,
    stage: Usd.Stage,
) -> Usd.Prim:
    """Create material with rigid-body physics properties.

    Different from the IsaacLab version, :func:`isaaclab.sim.spawners.materials.spawn_spawn_rigid_body_material`
    this function allows users to specify the stage where the material is spawned.

    Rigid body materials are used to define the physical properties to meshes of a rigid body. These
    include the friction, restitution, and their respective combination modes. For more information on
    rigid body material, please refer to the `documentation on PxMaterial <https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/classPxBaseMaterial.html>`_.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the physics material.

    Returns:
        The spawned rigid body material prim.

    Raises:
        ValueError:  When a prim already exists at the specified prim path and is not a material.
    """
    # create material prim if no prim exists
    if not prim_utils.is_prim_path_valid(prim_path):
        _ = UsdShade.Material.Define(stage, prim_path)

    # obtain prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is a material
    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A prim already exists at path: '{prim_path}' but is not a material.")
    # retrieve the USD rigid-body api
    usd_physics_material_api = UsdPhysics.MaterialAPI(prim)  # type: ignore
    if not usd_physics_material_api:
        usd_physics_material_api = UsdPhysics.MaterialAPI.Apply(prim)  # type: ignore
    # retrieve the collision api
    physx_material_api = PhysxSchema.PhysxMaterialAPI(prim)
    if not physx_material_api:
        physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(prim)

    # convert to dict
    cfg = cfg.to_dict()  # type: ignore
    del cfg["func"]  # type: ignore
    # set into USD API
    for attr_name in ["static_friction", "dynamic_friction", "restitution"]:
        value = cfg.pop(attr_name, None)  # type: ignore
        safe_set_attribute_on_usd_schema(usd_physics_material_api, attr_name, value, camel_case=True)
    # set into PhysX API
    for attr_name, value in cfg.items():  # type: ignore
        safe_set_attribute_on_usd_schema(physx_material_api, attr_name, value, camel_case=True)
    # return the prim
    return prim


@clone
def stage_specific_spawn_deformable_body_material(
    prim_path: str,
    cfg: physics_materials_cfg.StageSpecificDeformableBodyMaterialCfg,
    stage: Usd.Stage,
) -> Usd.Prim:
    """Create material with deformable-body physics properties.

    Different from the IsaacLab version, :func:`isaaclab.sim.spawners.materials.spawn_spawn_deformable_body_material`
    this function allows users to specify the stage where the material is spawned.

    Deformable body materials are used to define the physical properties to meshes of a deformable body. These
    include the friction and deformable body properties. For more information on deformable body material,
    please refer to the documentation on `PxFEMSoftBodyMaterial`_.

    .. note::
        This function is decorated with :func:`clone` that resolves prim path into list of paths
        if the input prim path is a regex pattern. This is done to support spawning multiple assets
        from a single and cloning the USD prim at the given path expression.

    Args:
        prim_path: The prim path or pattern to spawn the asset at. If the prim path is a regex pattern,
            then the asset is spawned at all the matching prim paths.
        cfg: The configuration for the physics material.

    Returns:
        The spawned deformable body material prim.

    Raises:
        ValueError:  When a prim already exists at the specified prim path and is not a material.

    .. _PxFEMSoftBodyMaterial: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/_api_build/structPxFEMSoftBodyMaterialModel.html
    """
    # create material prim if no prim exists
    if not prim_utils.is_prim_path_valid(prim_path):
        _ = UsdShade.Material.Define(stage, prim_path)

    # obtain prim
    prim = stage.GetPrimAtPath(prim_path)
    # check if prim is a material
    if not prim.IsA(UsdShade.Material):
        raise ValueError(f"A prim already exists at path: '{prim_path}' but is not a material.")
    # retrieve the deformable-body api
    physx_deformable_body_material_api = PhysxSchema.PhysxDeformableBodyMaterialAPI(prim)
    if not physx_deformable_body_material_api:
        physx_deformable_body_material_api = PhysxSchema.PhysxDeformableBodyMaterialAPI.Apply(prim)

    # convert to dict
    cfg = cfg.to_dict()  # type: ignore
    del cfg["func"]  # type: ignore
    # set into PhysX API
    for attr_name, value in cfg.items():  # type: ignore
        safe_set_attribute_on_usd_schema(physx_deformable_body_material_api, attr_name, value, camel_case=True)
    # return the prim
    return prim
